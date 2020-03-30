# DeepTract
The official implementation of the method proposed in Benou and Riklin-Raviv "DeepTract: A Probabilistic Deep Learning Framework for White Matter Fiber Tractography" https://arxiv.org/abs/1812.05129. 

----------
![Alt text](tracking_examples/DeepTract1.png?raw=true "Title") ![Alt text](tracking_examples/DeepTract2.png?raw=true "Title")
----------
If you find this code useful in your research or publication, please cite the paper:
```
@inproceedings{benou2019deeptract,
  title={Deeptract: A probabilistic deep learning framework for white matter fiber tractography},
  author={Benou, Itay and Raviv, Tammy Riklin},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={626--635},
  year={2019},
  organization={Springer}
}
```

## Usage
1) Clone/download repo.
2) Edit the config.py file to configure the parameters according to the desired usage. Follow the comments above each parameter.
3) Arrange the data: the DeepTract script expects a folder named "data" with the following folders/files structure:
```
   data
       --- dwi
          --- <dwi_file>.nii
          --- <bvecs_file>.bvecs
          --- <bvals_file>.bvals
       --- labels
          --- <tractography_file>.trk
       --- mask
          --- <brain_mask_file>.nii
       --- wm_mask
          --- <white_matter_mask>.nii
```
4) For training a new DeepTract model, run:
```
deeptract.py --train
```
For running streamline tractography using a trained DeepTract model, run:
```
deeptract.py --track
```
