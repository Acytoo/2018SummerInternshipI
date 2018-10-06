## 2018 Summer Internship I

### This repository is basically a record of our learning process of 2018 Summer Internship I, including data crawling, image processing, feature extraction and svm training.

  After learnig and trying, we built an image classfier, the classifier was uploaded to an [online judge](http://219.216.65.165:9000), the oj trained classfiers from all 21 teams on a dataset we didn't know, then tested them. The good news was that we were the top 3 of all the 21 teams(and we are the fastest group with least memory usage), we are team16, and full results can be found [here](http://219.216.65.165:9000/result).

![image](https://acytoo.github.io/HPSRC/2018Internship/result.png)

  When we built the classfier, we used this [dataset](https://drive.google.com/file/d/1DJPH0MGSi2t6OjEnvAXsP_WZMlaAfPgq/view?usp=sharing) to train, it was collected by 21 teams, each team chose one kind, all the 21 types were bear, bicycle, bird, car, cow, elk, fox, giraffe, horse, koala, lion, monkey, plane, puppy, sheep, statue, tiger, tower, train, whale and zebra. One funny thing was that at the begining, our model couldn't draw a distinction between tigers and zebras, it turned out that we were using greyscale, and, tiger stripes and zebra stripes are similiar in greyscale:). Later, we used cv2.split() to split r, g, b, and then put them in one array to extracted features. This way was much better.

  For this dataset, since it contains all kinds of forms, not one particular way of feature extractions can perform well on all the 21 kinds of images, we have tried  many methods, including LBP, HOG, SIFT, SURF and their combinations, finally we use LBP+HOG, each of them placed 50% of the total weight, we thought this way was balanced between speed and preciseness.

  The uploaded classifier is in [compepition_interface](https://github.com/Acytoo/2018summer/tree/master/compepition_interface) folder.



### Want something more? Go to [part II](https://github.com/Acytoo/baidu_ai_competition)
