# MPF-GAN: an enhanced architecture for 3D face reconstruction

**Authors:**  
Mehdi Malah, Fayçal Abbas, Ramzi Agaba, Dalal Bardou, Mohamed Chaouki Babahenini  

**Published in:**  
Multimedia Tools and Applications, 2024  
Springer

## Abstract
[Provide a brief abstract of your paper here.]

## Code Availability
In this article, we address the challenge of accurate 3D face reconstruction by proposing an enhanced architecture. We observe that using a single error calculation formula for the entire face leads to precise reconstruction of certain parts at the expense of others, particularly the eyes and mouth. We introduce a multi-task learning approach to address this issue, allowing our model to train on multiple data types. We propose a new analytical solution for mouth shape identification and its integration into the loss function optimization process. This solution resolves a problem that was often difficult to model in previous works accurately and has demonstrated improved reconstruction performance. In our research, we have also created and made available a new dataset for 3D face reconstruction. This dataset is unique as it focuses on different facial parts. The results show that our multi-task learning approach significantly improves the loss function for delicate facial parts, thus enhancing the quality of reconstruction in these specific areas. The code and data are publicly available on the project’s link: https://mpf-gan.github.io

## Citation
If you would like to cite this work, please use the following format:

```bibtex
@article{malah2024mpf,
  title={MPF-GAN: an enhanced architecture for 3D face reconstruction},
  author={Malah, Mehdi and Abbas, Fay{\c{c}}al and Agaba, Ramzi and Bardou, Dalal and Babahenini, Mohamed Chaouki},
  journal={Multimedia Tools and Applications},
  pages={1--18},
  year={2024},
  publisher={Springer}
}
