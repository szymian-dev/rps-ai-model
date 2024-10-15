import shutil
import os
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf

def split_dataset(classes : [str], src_dir : str, dest_dir : str, split_proportions : [float] = [0.8, 0.1, 0.1]) -> None:
    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    test_dir = os.path.join(dest_dir, 'test')
    
    for class_name in classes:
        class_path = os.path.join(src_dir, class_name)
        files = os.listdir(class_path)
        
        if len(files) == 0:
            raise Exception(f'No files found in {class_path}')

        train_files, test_val_files = train_test_split(files, test_size=(1 - split_proportions[0]))
        val_files, test_files = train_test_split(test_val_files, test_size=(split_proportions[2] / (split_proportions[1] + split_proportions[2])))

        _copy_files(class_path, os.path.join(train_dir, class_name), train_files)
        _copy_files(class_path, os.path.join(val_dir, class_name), val_files)
        _copy_files(class_path, os.path.join(test_dir, class_name), test_files)
        
def _copy_files(src_dir : str, dest_dir : str, files : [str]) -> None:
        os.makedirs(dest_dir, exist_ok=True)
        for file_name in files:
            shutil.copy(os.path.join(src_dir, file_name), os.path.join(dest_dir, file_name))
            
def check_for_duplicates_in_dataset(src_dir : str) -> None:
    files = []
    for root, _, filenames in os.walk(src_dir):
        for filename in filenames:
            files.append(filename)
    if len(files) != len(set(files)):
        raise Exception('Duplicates found in dataset')
    else:
        print('No duplicates found in dataset')
        
    print(len(files), 'files found in dataset')
    
def save_model(model, model_name: str, models_dir: str, history: dict = None) -> None:
    model_path = os.path.join(models_dir, model_name + '.h5')
    
    counter = 1
    unique_model_name = model_name 

    while os.path.exists(model_path):
        model_path = os.path.join(models_dir, f"{model_name}_{counter}.h5")
        counter += 1
    
    model.save(model_path)
    print(f'Model saved in {model_path}')
    
    if counter == 1:
        history_path = os.path.join(models_dir, f"{model_name}_history.pkl")
    else:
        history_path = os.path.join(models_dir, f"{model_name}_{counter - 1}_history.pkl")
    
    if history is not None:
        with open(history_path, 'wb') as f:
            pickle.dump(history, f)
        print(f'History saved in {history_path}')
        
def load_history(model_name: str, models_dir: str) -> dict:
    history_path = os.path.join(models_dir, model_name + '_history.pkl')
    if not os.path.exists(history_path):
        raise Exception(f'History file {history_path} not found')
    
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    
    return history

def load_model(model_name: str, models_dir: str) -> tf.keras.models.Model:
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path):
        raise Exception(f'Model file {model_path} not found')
    
    model = tf.keras.models.load_model(model_path)
    return model

def compile_and_train_model(create_model_func, create_model_args: dict, optimizer, loss, metrics, train_generator, val_generator, epochs: int, models_dir: str, model_name: str, verbose: int = 1) -> dict:
    model = create_model_func(**create_model_args)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return train_model(model, train_generator, val_generator, epochs, models_dir, model_name, verbose)

def train_model(model, train_generator, val_generator, epochs: int, models_dir: str, model_name: str, verbose: int = 1) -> dict:
    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, verbose=verbose)
    save_model(model, model_name, models_dir, history.history)
    return history.history
       
