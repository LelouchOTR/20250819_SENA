# List of generic biomedical terms to filter out
GENERIC_BIOMEDICAL_TERMS = {
    # Basic biological terms
    'cell', 'cells', 'tissue', 'tissues', 'organ', 'organs',
    'protein', 'proteins', 'gene', 'genes', 'dna', 'rna',
    'mrna', 'transcription', 'translation', 'expression',
    
    # Cellular components
    'membrane', 'nucleus', 'cytoplasm', 'mitochondria', 'mitochondrion',
    'chloroplast', 'ribosome', 'endoplasmic', 'reticulum', 'golgi',
    'lysosome', 'peroxisome', 'cytoskeleton', 'centrosome',
    
    # Biological processes
    'process', 'activity', 'function', 'functions', 'pathway', 'pathways',
    'mechanism', 'mechanisms', 'regulation', 'regulatory', 'control',
    'signaling', 'signalling', 'signal', 'response', 'responses',
    
    # Molecular biology terms
    'binding', 'site', 'sites', 'domain', 'domains', 'motif', 'motifs',
    'sequence', 'sequences', 'structure', 'structural',
    'enzyme', 'enzymes', 'catalytic', 'catalysis',
    
    # Cellular processes
    'cellular', 'metabolic', 'biosynthesis', 'synthesis', 'degradation',
    'catabolism', 'anabolism', 'metabolism', 'homeostasis',
    
    # General descriptors
    'general', 'specific', 'type', 'types', 'class', 'classes',
    'family', 'families', 'group', 'groups', 'component', 'components',
    
    # Biological systems
    'system', 'systems', 'network', 'networks', 'complex', 'complexes',
    
    # Temporal terms
    'development', 'developmental', 'growth', 'differentiation',
    'maturation', 'aging', 'senescence',
    
    # Quality descriptors
    'positive', 'negative', 'increased', 'decreased', 'reduced',
    'elevated', 'activated', 'inhibited', 'regulated',
    
    # Location terms
    'nuclear', 'cytoplasmic', 'membranous', 'extracellular',
    'intracellular', 'cell surface', 'peripheral',
    
    # Common prefixes/suffixes
    'bio', 'biochemical', 'molecular', 'genetic', 'genomic',
    'proteomic', 'cellular', 'physiological',
    
    # Generic process terms
    'process', 'processes', 'processing', 'production', 'assembly',
    'organization', 'organisation', 'maintenance', 'modification',
    'transport', 'localization', 'localisation', 'movement',
    
    # Generic molecular terms
    'molecule', 'molecules', 'compound', 'compounds', 'substance',
    'entity', 'entities', 'factor', 'factors', 'element', 'elements',
    
    # Action verbs often in BP names
    'involved', 'associated', 'related', 'mediated', 'driven',
    'dependent', 'independent', 'induced', 'repressed',
    
    # Category 1: Extremely General Processes
    # (These are often the highest-level parent terms in gene ontology hierarchies)
    'biosynthetic', 'catabolic', 'metabolic', 'morphogenesis',
    'secretion', 'stimulus', 'proliferation', 'inhibition',
    
    # Category 2: Vague Actions & Generic Nouns
    # (These words are meaningless without knowing the object or context)
    'formation', 'generation', 'transition', 'transmission',
    'migration', 'contraction', 'junction', 'cycle',
    
    # Category 3: Fragmented Chemical & Component Terms
    # (These are parts of more specific multi-word terms and should not appear alone)
    'acid', 'actin', 'fatty', 'filament', 'fluid',
    'hormone', 'ion', 'organic', 'phosphate',
    'receptor', 'vitamin',
    
    # Category 4: Overly Broad Anatomical Terms
    # (These are usually too high-level unless your entire study is a comparison between them)
    'blood', 'embryonic', 'epithelial', 'muscle',
    'nervous', 'skeletal', 'systemic', 'vessel'
}