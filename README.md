# Avatar Tools
Version 1.1

Supervised machine learning is the process of using past experience to
predict the future. "Ensembles" are a machine-learning meta-method
that can be applied to most machine learning algorithms. Ensembles
generally greatly improve accuracy, reduce or remove most of the
design issues presented by machine learning, and are admirably suited
to parallel and distributed computation.

The Avatar Tools codes are an implementation of ensembles specifically
for decision trees.

Some features that distinguish Avatar Tools from other "ensembles for
decision trees" codes are:

* Does the bookkeeping necessary for out of bag (OOB) validation.

* Can use OOB validation to automatically determine optimal ensemble size.

* Provides an MPI-based parallel implementation, for distributed
  operation.

* Provides convenient tools for cross-validation, to assess the
  accuracy provided by a training set.

* Handles both plain text and Exodus simulation data.


The Avatar Tools codes are intended for Unix machines, and are known
to build and pass their tests on Linux, Mac OS X, and Solaris
machines.

See INSTALL for installation instructions,

See the files in support/ for sample data and a brief tutorial.

## Citations

Please cite one or more of the following papers, if you wish to cite the Avatar Tools:

```
@INPROCEEDINGS{Chawla2,
  author = {Nitesh Chawla and Thomas Moore and Kevin Bowyer and Lawrence Hall
        and Clayton Springer and Philip Kegelmeyer},
  title = {Bagging is a Small-Data-Set Phenomenon},
  booktitle = {International Conference on Computer Vision and Pattern Recognition
        (CVPR)},
  year = {2001}
}

@ARTICLE{CaBoHaKe02,
  author = {Nitesh V.~Chawla and Kevin W.~Bowyer and Lawrence O.~Hall and W.~Philip
        Kegelmeyer},
  title = {{SMOTE}: Synthetic Minority Over-sampling Technique},
  journal = {Journal of Artificial Intelligence Research},
  year = {2002},
  volume = {16},
  pages = {321-357},
  url = {http://adsabs.harvard.edu/abs/2011arXiv1106.1813B},
}

ARTICLE{ChHaBoKe04,
  author = {Nitesh V.~Chawla and Lawrence O.~Hall and Kevin W.~Bowyer and W.~Philip
        Kegelmeyer },
  title = {Learning ensembles from bites: A scalable and accurate approach},
  journal = {Journal of Machine Learning Research},
  year = {2004},
  volume = {5},
  pages = {421--451}
}

@INPROCEEDINGS{Chawla4,
  author = {Nitesh V.~Chawla and Lawrence O.~Hall and Kevin W.~Bowyer and Thomas
        E.~Moore and W.~Philip Kegelmeyer},
  title = {Distributed Pasting of Small Votes},
  booktitle = {International Workshop on Multiple Classifier Systems},
  year = {2002},
  address = {Sardegna, Italy},
  month = {June}
}

TECHREPORT{Hall2,
  author = {L.O.~Hall and K.W.~Bowyer and N.~Chawla and T.~Moore and W.~Philip
        Kegelmeyer},
  title = {{AVATAR} --- Adaptive Visualization Aid for Touring and Recovery},
  institution = {Sandia National Laboratories},
  year = {2000},
  type = {Sandia Report},
  number = {SAND2000-8203},
  month = {January}
}

@article {CiHoCh11,
   author = {Cieslak, David and Hoens, T. and Chawla, Nitesh and Kegelmeyer, W.},
   title = {Hellinger distance decision trees are robust and skew-insensitive},
   journal = {Data Mining and Knowledge Discovery},
   publisher = {Springer Netherlands},
   issn = {1384-5810},
   vol = {24}, 
   issue = 1,
   pages = {136--158},
   url = {http://dx.doi.org/10.1007/s10618-011-0222-1},
   note = {10.1007/s10618-011-0222-1},
   year = {2012}
}
```


## Acknowledgments

The Avatar Tools come from a decade-long machine learning research
program led by Philip Kegelmeyer at Sandia National Laboratories, with
contributions from collaborators at the University of South Florida
(notably, Professor Larry Hall and students Robert Banfield and Larry
Shoemaker) and at Notre Dame (Computer Science Department Chair Kevin
Bower and Professor Nitesh Chawla.)

Early research implementations largely came from Robert Banfield and
Steven Eschrich, at USF. The current production implementation is by
Ken Buch, a contractor with Limit Point Systems.


If you have questions, contact:

Philip Kegelmeyer, wpk@sandia.gov

