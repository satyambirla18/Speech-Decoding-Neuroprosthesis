import os
import h5py
import pandas as pd

def extract_info(base_dir):
    
    all_info = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.hdf5'):
                file_path = os.path.join(root, file)
                try:
                    with h5py.File(file_path, 'r') as f:
                        keys = list(f.keys())
                        num_trials = len(keys)
                        sessions = []
                        n_time_steps_list = []
                        seq_len_list = []
                        block_nums = []
                        trial_nums = []
                        first_trans = None
                        first_shape = None
                        first_seq_class_ids = None
                        first_sentence_label = None
                        
                        for i, key in enumerate(keys):
                            g = f[key]
                            sessions.append(g.attrs['session'])
                            n_time_steps_list.append(g.attrs['n_time_steps'])
                            seq_len_list.append(g.attrs['seq_len'] if 'seq_len' in g.attrs else None)
                            block_nums.append(g.attrs['block_num'])
                            trial_nums.append(g.attrs['trial_num'])
                            
                            if i == 0:  
                                neural_features_shape = g['input_features'].shape
                                first_shape = neural_features_shape
                                if 'transcription' in g:
                                    trans_array = g['transcription'][:]
                                    first_trans = ''.join(chr(int(c)) for c in trans_array if c != 0).strip()
                                if 'seq_class_ids' in g:
                                    first_seq_class_ids = list(g['seq_class_ids'][:])
                                if 'sentence_label' in g.attrs:
                                    first_sentence_label = g.attrs['sentence_label']
                        
                        sessions = list(set(sessions))
                        info = {
                            'file_path': file_path,
                            'num_trials': num_trials,
                            'sessions': sessions,
                            'min_n_time_steps': min(n_time_steps_list),
                            'max_n_time_steps': max(n_time_steps_list),
                            'min_seq_len': min([sl for sl in seq_len_list if sl is not None]) if any(sl is not None for sl in seq_len_list) else None,
                            'max_seq_len': max([sl for sl in seq_len_list if sl is not None]) if any(sl is not None for sl in seq_len_list) else None,
                            'first_neural_shape': str(first_shape),
                            'first_transcription': first_trans,
                            'first_seq_class_ids': first_seq_class_ids,
                            'first_sentence_label': first_sentence_label,
                        }
                        all_info.append(info)
                except Exception as e:
                    print(f"Error in {file_path}: {e}")
    
    df = pd.DataFrame(all_info)
    print(f"Total files: {len(all_info)}")
    df.to_csv('hdf5_data_summary.csv', index=False)

if __name__ == "__main__":
    extract_info(base_dir = 'hdf5_data_final')