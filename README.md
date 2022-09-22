## About

Neural Architecture Search pipeline framework aim to provide a base start for anyone who intend to use NAS. 

### Project Team
* Stephen McGough - ([stephen.mcgough@newcastle.ac.uk](mailto:stephen.mcgough@newcastle.ac.uk))
* Rob Geada - ([rob@geada.net](mailto:rob@geada.net))
* David Towers - ([d.towers2@newcastle.ac.uk](mailto:d.towers2@newcastle.ac.uk))
* Amir Atapour-Abarghouei - ([amir.atapour-abarghouei@durham.ac.uk](mailto:amir.atapour-abarghouei@durham.ac.uk))


### RSE Contact
Nik Khadijah Nik Aznan
RSE Team  
Newcastle University  
([nik.nik-aznan@newcastle.ac.uk](mailto:nik.nik-aznan@newcastle.ac.uk))  

### Built With

[Python=3.10.7](https://something.com)  
[Pytorch=1.12.1](https://pytorch.org)  

### How to use
The framework is ready to use with very little modification. 
Input format - you need to choose your input format and in the `main_framework.py` you need to specify the `data_info`. You also can change the `augment_style`.

```python  
data_process = Data_Pipeline(
        data_info=f"{Path.home()}/Data/small_dataset",
        augment_style="flip",
        BATCHSIZE=28,
    )
```
    
`data_info` is depending on data type used. For `image_download`, `data_info` can be MNIST or fashion_mnist or any image dataset that can be downloaded from Pytorch. For `image_folder`, `data_info` is the path to the the image folder. For `numpy_image`, `data_info` is the path to the the numpy files.

The import in `data_pipeline` need to be changed accordingly. 
Use `from image_folder import Image_Folder` for the image folders, 
`from image_download import Image_Download` for the downloaded images and 
`from numpy_image import Numpy_Image` for the numpy files.

The `nas_pipeline` can be modified to add your own version of NAS by importing the NAS python file instead of `nas_search`.
