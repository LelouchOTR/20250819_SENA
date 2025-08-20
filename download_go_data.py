import os
import gzip
import urllib.request
from pathlib import Path

def download_file(url, output_path):
    """Download a file from a URL and save it locally"""
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Successfully downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False

def main():
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Download GO OBO file
    go_obo_url = "http://current.geneontology.org/ontology/go-basic.obo"
    go_obo_path = data_dir / 'go-basic.obo'
    
    if not go_obo_path.exists():
        if not download_file(go_obo_url, str(go_obo_path)):
            print("Warning: Could not download GO OBO file. Some functionality may be limited.")
    
    # Download gene2go file
    gene2go_url = "https://ftp.ncbi.nih.gov/gene/DATA/gene2go.gz"
    gene2go_gz_path = data_dir / 'gene2go.gz'
    gene2go_path = data_dir / 'gene2go'
    
    if not gene2go_path.exists():
        if gene2go_gz_path.exists() or download_file(gene2go_url, str(gene2go_gz_path)):
            print("Extracting gene2go file...")
            try:
                with gzip.open(str(gene2go_gz_path), 'rb') as f_in:
                    with open(str(gene2go_path), 'wb') as f_out:
                        f_out.write(f_in.read())
                print(f"Extracted to {gene2go_path}")
            except Exception as e:
                print(f"Error extracting gene2go file: {str(e)}")
    
    print("\nSetup complete!")
    print("You can now run the analysis pipeline.")

if __name__ == "__main__":
    main()
