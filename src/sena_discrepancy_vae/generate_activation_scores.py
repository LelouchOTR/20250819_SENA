import numpy as np
import pickle
import torch
import pandas as pd 
import os
from tqdm import tqdm
from collections import defaultdict
import numpy as np
import sys
import logging
import json
from model import CMVAE, NetworkActivity_layer
from utils import Norman2019DataLoader, Wessel2023HEK293DataLoader
import argparse
import glob

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_model_file(directory):
    """Find the model file in the directory"""
    # First check for best_model.pt
    best_model_path = os.path.join(directory, 'best_model.pt')
    if os.path.exists(best_model_path):
        return best_model_path
    
    # If best_model.pt doesn't exist, look for other .pt files
    pt_files = glob.glob(os.path.join(directory, '*.pt'))
    if pt_files:
        # Sort by modification time, newest first
        pt_files.sort(key=os.path.getmtime, reverse=True)
        logging.info(f"Found model files: {pt_files}")
        logging.info(f"Using the newest model file: {pt_files[0]}")
        return pt_files[0]
    
    return None

def generating_data(config_file, fpath, batch_size=32):
    """Generate activation scores using config file approach"""
    
    ## detect device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # define parameters
    dataset_name = config_file["dataset_name"]

    logging.info(f"Loading {dataset_name} dataset")
    if 'Norman2019' in dataset_name:
        data_handler = Norman2019DataLoader(batch_size=batch_size)
        data_handler.gene_var = "guide_ids"
    elif dataset_name == 'wessel_hefk293':
        data_handler = Wessel2023HEK293DataLoader(batch_size=batch_size)
        data_handler.gene_var = 'TargetGenes'

    #load data and reset index
    adata = data_handler.adata
    adata.obs = adata.obs.reset_index(drop=True)
    ptb_targets = data_handler.ptb_targets
    gos = data_handler.gos

    """build pert idx dict"""
    idx_dict = {}
    for knockout in ['ctrl'] + ptb_targets:
        if knockout == 'ctrl':
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == '']).index.values
        else:
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == knockout]).index.values

    # Debug: print first few perturbation targets
    logging.info(f"Found {len(ptb_targets)} perturbation targets")
    logging.info(f"First few targets: {ptb_targets[:5]}")
    logging.info(f"First few GO terms: {gos[:5] if gos else 'None'}")

    """load best model"""
    #load weights
    model_path = find_model_file(fpath)
    if model_path is None:
        raise FileNotFoundError(f"No model files found in {fpath}")
    
    logging.info(f"Loading model from {model_path}")
    model = torch.load(model_path, map_location='cpu')
    
    # Debug: print model structure
    logging.info(f"Model type: {type(model)}")
    if hasattr(model, '__dict__'):
        logging.info(f"Model attributes: {list(vars(model).keys())}")
    if hasattr(model, 'G'):
        logging.info(f"Causal graph G shape: {model.G.shape if model.G is not None else 'None'}")

    ##
    n_pertb = len(ptb_targets)
    pert_dict = {}
    info_dict = defaultdict(lambda: defaultdict(list))
    results_dict = {}


    """compute"""
    with torch.no_grad():

        for gene in tqdm(idx_dict, desc='generating activity score for perturbations'):
            
            idx = idx_dict[gene]
            mat = torch.from_numpy(adata.X[idx,:].todense()).to(device).double()

            """first layer"""

            na_score_fc1 = model.fc1(mat)
            info_dict['fc1'][gene].append(na_score_fc1.detach().cpu().numpy())

            """mean + var"""

            na_score_fc_mean = model.fc_mean(na_score_fc1)
            info_dict['fc_mean'][gene].append(na_score_fc_mean.detach().cpu().numpy())

            na_score_fc_var = torch.nn.Softplus()(model.fc_var(na_score_fc1))
            info_dict['fc_var'][gene].append(na_score_fc_var.detach().cpu().numpy())

            """reparametrization trick"""

            na_score_mu, na_score_var = model.encode(mat)
            na_score_z = model.reparametrize(na_score_mu, na_score_var)
            info_dict['z'][gene].append(na_score_z.detach().cpu().numpy())

            """causal graph"""

            if gene != 'ctrl':

                #define ptb idx
                ptb_idx = np.where(np.array(ptb_targets)==gene)[0][0]

                #generate one-hot-encoder
                c = torch.zeros(size=(1,n_pertb))
                c[:,ptb_idx] = 1
                c = c.to(device).double()

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=1)
                bc2, csz2 = bc, csz
                info_dict['bc_temp1'][gene].append(bc.detach().cpu().numpy())

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=100)
                bc2, csz2 = bc, csz
                info_dict['bc_temp100'][gene].append(bc.detach().cpu().numpy())

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=1000)
                bc2, csz2 = bc, csz
                info_dict['bc_temp1000'][gene].append(bc.detach().cpu().numpy())

                # compute assignation
                if ptb_idx not in pert_dict:
                    pert_dict[ptb_idx] = bc[0].argmax().__int__()

                #interventional U
                na_score_u = model.dag(na_score_z, bc, csz, bc2, csz2, num_interv=1)
                info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

            else:

                #observational U
                na_score_u = model.dag(na_score_z, 0, 0, 0, 0, num_interv=0)
                info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

    """build dataframes within each category"""
    for layer in info_dict:

        temp_df = []
        for gene in info_dict[layer]:
            info_dict[layer][gene] = pd.DataFrame(np.vstack(info_dict[layer][gene]))
            info_dict[layer][gene].index = [gene] * info_dict[layer][gene].shape[0]
            temp_df.append(info_dict[layer][gene])
        
        #substitute
        results_dict[layer] = pd.concat(temp_df)
        if layer == 'fc1':
            results_dict[layer].columns = gos

    #add pertb_dict
    results_dict['pert_map'] = pd.DataFrame(pert_dict, index=[0]).T
    results_dict['pert_map'].columns = ['c_enc_mapping']
    
    # Debug: print causal graph info
    if hasattr(model, 'G'):
        results_dict['causal_graph'] = model.G.detach().cpu().numpy()
        logging.info(f"Causal graph extracted with shape: {results_dict['causal_graph'].shape}")
        # Print first few elements of causal graph
        cg = results_dict['causal_graph']
        logging.info(f"First few elements of causal graph:\n{cg[:5, :5] if cg.shape[0] >= 5 and cg.shape[1] >= 5 else cg}")
    else:
        logging.warning("Model does not have attribute 'G' for causal graph")
        results_dict['causal_graph'] = None

    """add weights layers (delta) for """
    results_dict['mean_delta_matrix'] = pd.DataFrame(model.fc_mean.weight.detach().cpu().numpy().T, index=gos) 
    results_dict['std_delta_matrix'] = pd.DataFrame(model.fc_var.weight.detach().cpu().numpy().T, index=gos) 

    """save info"""
    output_path = os.path.join(fpath, 'activation_scores.pickle')
    logging.info(f"Saving activation scores to {output_path}")
    with open(output_path, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Debug: print what was saved
    logging.info(f"Saved keys in results_dict: {list(results_dict.keys())}")
    for key, value in results_dict.items():
        if hasattr(value, 'shape'):
            logging.info(f"  {key}: shape {value.shape}")
        else:
            logging.info(f"  {key}: type {type(value)}")


def generating_data_from_model_path(model_path, batch_size=32):
    """Generate activation scores directly from model path"""
    
    logging.info(f"Loading model from {model_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load model
    model_data = torch.load(model_path, map_location='cpu')
    logging.info(f"Model type: {type(model_data)}")
    
    # Try to determine dataset type from model structure
    dataset_name = "Norman2019"
    logging.info(f"Assuming dataset: {dataset_name}")
    
    # Load data
    data_handler = Norman2019DataLoader(batch_size=batch_size)
    data_handler.gene_var = "guide_ids"
    
    # Load data and reset index
    adata = data_handler.adata
    adata.obs = adata.obs.reset_index(drop=True)
    ptb_targets = data_handler.ptb_targets
    gos = data_handler.gos
    
    logging.info(f"Found {len(ptb_targets)} perturbation targets")
    logging.info(f"First few targets: {ptb_targets[:5]}")
    logging.info(f"First few GO terms: {gos[:5] if gos else 'None'}")
    
    # Handle model data (could be model object or tuple)
    if isinstance(model_data, tuple):
        model = model_data[0]  # Assume first element is the model
        logging.info("Model data is a tuple, using first element")
    else:
        model = model_data
    
    # Debug: print model structure
    logging.info(f"Model type: {type(model)}")
    if hasattr(model, '__dict__'):
        logging.info(f"Model attributes: {list(vars(model).keys())}")
    if hasattr(model, 'G'):
        logging.info(f"Causal graph G shape: {model.G.shape if model.G is not None else 'None'}")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Build pert idx dict
    idx_dict = {}
    for knockout in ['ctrl'] + ptb_targets:
        if knockout == 'ctrl':
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == '']).index.values
        else:
            idx_dict[knockout] = (adata.obs[adata.obs[data_handler.gene_var] == knockout]).index.values

    n_pertb = len(ptb_targets)
    pert_dict = {}
    info_dict = defaultdict(lambda: defaultdict(list))
    results_dict = {}

    """compute"""
    with torch.no_grad():
        for gene in tqdm(idx_dict, desc='generating activity score for perturbations'):
            idx = idx_dict[gene]
            mat = torch.from_numpy(adata.X[idx,:].todense()).to(device).double()

            """first layer"""
            na_score_fc1 = model.fc1(mat)
            info_dict['fc1'][gene].append(na_score_fc1.detach().cpu().numpy())

            """mean + var"""
            na_score_fc_mean = model.fc_mean(na_score_fc1)
            info_dict['fc_mean'][gene].append(na_score_fc_mean.detach().cpu().numpy())

            na_score_fc_var = torch.nn.Softplus()(model.fc_var(na_score_fc1))
            info_dict['fc_var'][gene].append(na_score_fc_var.detach().cpu().numpy())

            """reparametrization trick"""
            na_score_mu, na_score_var = model.encode(mat)
            na_score_z = model.reparametrize(na_score_mu, na_score_var)
            info_dict['z'][gene].append(na_score_z.detach().cpu().numpy())

            """causal graph"""
            if gene != 'ctrl':
                #define ptb idx
                ptb_idx = np.where(np.array(ptb_targets)==gene)[0][0]

                #generate one-hot-encoder
                c = torch.zeros(size=(1,n_pertb))
                c[:,ptb_idx] = 1
                c = c.to(device).double()

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=1)
                bc2, csz2 = bc, csz
                info_dict['bc_temp1'][gene].append(bc.detach().cpu().numpy())

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=100)
                bc2, csz2 = bc, csz
                info_dict['bc_temp100'][gene].append(bc.detach().cpu().numpy())

                # decode an interventional sample from an observational sample    
                bc, csz = model.c_encode(c, temp=1000)
                bc2, csz2 = bc, csz
                info_dict['bc_temp1000'][gene].append(bc.detach().cpu().numpy())

                # compute assignation
                if ptb_idx not in pert_dict:
                    pert_dict[ptb_idx] = bc[0].argmax().__int__()

                #interventional U
                na_score_u = model.dag(na_score_z, bc, csz, bc2, csz2, num_interv=1)
                info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

            else:
                #observational U
                na_score_u = model.dag(na_score_z, 0, 0, 0, 0, num_interv=0)
                info_dict['u'][gene].append(na_score_u.detach().cpu().numpy())

    """build dataframes within each category"""
    for layer in info_dict:
        temp_df = []
        for gene in info_dict[layer]:
            info_dict[layer][gene] = pd.DataFrame(np.vstack(info_dict[layer][gene]))
            info_dict[layer][gene].index = [gene] * info_dict[layer][gene].shape[0]
            temp_df.append(info_dict[layer][gene])
        
        #substitute
        results_dict[layer] = pd.concat(temp_df)
        if layer == 'fc1':
            results_dict[layer].columns = gos

    #add pertb_dict
    results_dict['pert_map'] = pd.DataFrame(pert_dict, index=[0]).T
    results_dict['pert_map'].columns = ['c_enc_mapping']
    
    # Debug: print causal graph info
    if hasattr(model, 'G'):
        results_dict['causal_graph'] = model.G.detach().cpu().numpy()
        logging.info(f"Causal graph extracted with shape: {results_dict['causal_graph'].shape}")
        # Print first few elements of causal graph
        cg = results_dict['causal_graph']
        logging.info(f"First few elements of causal graph:\n{cg[:5, :5] if cg.shape[0] >= 5 and cg.shape[1] >= 5 else cg}")
    else:
        logging.warning("Model does not have attribute 'G' for causal graph")
        results_dict['causal_graph'] = None

    """add weights layers (delta) for """
    results_dict['mean_delta_matrix'] = pd.DataFrame(model.fc_mean.weight.detach().cpu().numpy().T, index=gos) 
    results_dict['std_delta_matrix'] = pd.DataFrame(model.fc_var.weight.detach().cpu().numpy().T, index=gos) 

    """save info"""
    # Save to the same directory as the model
    output_dir = os.path.dirname(model_path)
    output_path = os.path.join(output_dir, 'activation_scores.pickle')
    logging.info(f"Saving activation scores to {output_path}")
    with open(output_path, 'wb') as handle:
        pickle.dump(results_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Debug: print what was saved
    logging.info(f"Saved keys in results_dict: {list(results_dict.keys())}")
    for key, value in results_dict.items():
        if hasattr(value, 'shape'):
            logging.info(f"  {key}: shape {value.shape}")
        else:
            logging.info(f"  {key}: type {type(value)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process configuration and generate data.")
    parser.add_argument("folder_name", type=str, nargs='?', default="example", 
                       help="The name of the folder in the results directory OR path to model file")
    args = parser.parse_args()

    # Check if the argument is a direct path to a model file
    if args.folder_name.endswith('.pt') and os.path.exists(args.folder_name):
        logging.info(f"Direct model file path detected: {args.folder_name}")
        try:
            generating_data_from_model_path(args.folder_name)
        except Exception as e:
            logging.error(f"Error generating data from model path: {e}")
            import traceback
            traceback.print_exc()
    else:
        logging.info(f"Folder name {args.folder_name} selected")
        
        # Define fpath
        fpath = os.path.join(os.getcwd(), 'results', args.folder_name)
        logging.info(f"Working directory: {fpath}")

        # Check if this is a valid results directory
        if not os.path.exists(fpath):
            # Try looking in pretrained_models directory
            pretrained_path = os.path.join(os.getcwd(), 'pretrained_models', args.folder_name)
            if os.path.exists(pretrained_path):
                fpath = pretrained_path
                logging.info(f"Found pretrained models directory: {fpath}")
            else:
                raise FileNotFoundError(f"Neither results directory nor pretrained_models directory found for {args.folder_name}")

        #get dataset_name
        config_path = os.path.join(fpath, 'config.json')
        logging.info(f"Loading config from {config_path}")
        
        # Check if config file exists
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found at {config_path}")
            logging.info("Attempting to generate activation scores without config...")
            # Try to generate data without config
            model_path = find_model_file(fpath)
            if model_path is None:
                raise FileNotFoundError(f"No model files found in {fpath}")
            
            try:
                generating_data_from_model_path(model_path)
            except Exception as e:
                logging.error(f"Error generating data: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
        else:
            with open(config_path, 'r') as file:
                config_file = json.load(file)
            
            # Debug: print config contents
            logging.info(f"Config contents: {config_file}")
            
            #generate pickle
            generating_data(config_file, fpath)
