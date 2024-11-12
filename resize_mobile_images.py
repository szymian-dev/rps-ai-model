from PIL import Image
import os

'''
    Script for resizing images in folders 'rock', 'paper', 'scissors' to new size.
    Images were taken in 4:3 aspect ratio, so new size is 400x300 in order to be somewhat similar to original dataset images (300x200).
'''

resampling_filter = Image.Resampling.LANCZOS # Resampling filter to use when resizing images

main_folder = '../images/aug_data_night/'  # Path to folder with subfolders 'rock', 'paper', 'scissors'
if not os.path.exists(main_folder):
    raise Exception(f"Folder {main_folder} does not exist")
    
output_folder = './resized_images/'  
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    
# New image size
new_size = (400, 300)


folders = ['rock', 'paper', 'scissors']

for folder in folders:
    folder_path = os.path.join(main_folder, folder)
    output_folder_path = os.path.join(output_folder, folder)
    
    files_affected = 0
    
    print(f"Processing folder {folder}...")
    print("Please be patient, this may take a while...")

    if not os.path.exists(folder_path):
        raise Exception(f"Folder {folder} does not exist")
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            with Image.open(file_path) as img:
                
                # Check if image is in vertical orientation
                if img.height > img.width:
                    img = img.rotate(90, expand=True)  # 90 degrees clockwise
                    
                img_resized = img.resize(new_size, resampling_filter)
                
                output_file_path = os.path.join(output_folder_path, filename)
                img_resized.save(output_file_path)
                files_affected += 1
    
    print(f"Folder {folder} done. Files affected: {files_affected}")

print("All done. :)")
