# GPU-Spatial Join amomg polygons

Small GIS shape files link: https://www.dropbox.com/sh/uikipog04hcuza9/AAA8a9uGqvxAYyXmyUGdRuEda?dl=0

Big GIS shap files link: https://www.dropbox.com/sh/5vs08se9ke1vsal/AACHXelB-ZCqBMiW9jYJfp1xa?dl=0

It is a Python implementation of a research article 
"GCMF: an efficient end-to-end spatial join system over large polygonal datasets on GPGPU platform" 
by Danial Aghajarian	(Georgia State University)
   Satish Puri (Marquette University)
   Sushil Prasad	(Georgia State University)
   Link: https://dl.acm.org/ft_gateway.cfm?id=2996982&ftid=1823205&dwn=1&CFID=6660507&CFTOKEN=2b565020174a2014-A8DCDDDE-D46D-C2EB-9980ED76EAED2925

As a first version of GCMF in Python, its performance is not as good as the results reported in the original paper. 
This can be explained by two main reasons: (1) Utilization of GPU; (2) my limited understanding of the GCMF. I developed 
a load-balanced version as well but it did not work as well as I expected. I will be turning this in C/C++ and updating on Github.

....
