# CS445_morphing_project

Dataset for face morphing:
300W: 
A dataset for facial landmark detection, with face images and labelled landmarks.
Raw face images can also be used for manual key point picking and morphing.
https://ibug.doc.ic.ac.uk/resources/300-W/

AFW:
similar to 300W, contains face images and labelled landmarks.
This dataset is more challenging, as multiple faces appear in one image.
May not be suitable for morphing, but good for training landmark detection models.
https://exposing.ai/afw/

Helen:
A larger dataset with single faces and annotations
https://www.kaggle.com/datasets/kmader/helen-eye-dataset

Facial Landmark Detection:
https://github.com/D-X-Y/landmark-detection



The facial landmark detection model used in the project is ADNet from Huang,Yangyu, et. al. The code is in "./ADNet" folder.
The dataset currently tested is the 300W dataset, which can be downloaded from the links above.
To prepare the data for training and testing, first organize the folder as follows:
First make a work_dir folder
---work_dir
    ---log
    ---model
        ---train.pkl
        ---train.onnx
    ---data
        ---alignment
            ---300w
                ---rawImages
                    ---afw
                    ---Bounding Boxes
                    ---helen
                    ---ibug
                    ---lfpw
                    ---compute_error.m
                ---test.tsv
                ---train.tsv

Dependency:
I ran with python3.10, pytorch with cuda 12.4. numpy version is 1.26.4 (current numpy verions are not compatable with the code), all other packages are installed directly using "pip install package_name".

To run testing, use the script:
python main.py --mode=test --config_name=alignment --pretrained_weight=path_to_train.pkl --device_ids=0 --work_dir=path_to_work_dir
Currently, running code requires a wandb account. Change wandb_key variable in "ADNet/base.py", line 99 to your key.
Changing batch size may be required so that the data fits in the GPU memory. change them in "ADNet/conf/alignment.py", line 17 and "ADNet/conf/base.py", line 40, 41, 42.


## Attribution

This project includes code from the [ADNet repository](https://github.com/huangyangyu/ADNet) by Yangyu Huang et al., used under the MIT License.

Original work:
Huang, Yangyu, et al. "ADNet: Leveraging Error-Bias Towards Normal Direction in Face Alignment." *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021, pp. 3080–3090.

MIT License © 2021 Yangyu Huang