## It is HIGHLY probable that IITP is guilty of overfitting while testing on the development set. 
## We get low scores when training on tr and testing on dev, but get high scores when testing on the goldtest set!

goldtest_nb, !wordembeddings, !containsURL =  0.25
goldtest_nb, !containsURL = 0.50
goldtest_nb =  0.50

goldtest_nb, !wordembeddings, user, reply = 0.50
goldtest_nb, !wordembeddings, reply = 0.53



goldtest_svm, !containsURL = 0.321
goldtest_svm = 0.321

goldtest_dt, !contansURL: 0.32
goldtest_dt: 0.28

dev_rnn = 0.32

goldtest_rnn, !wordembeddings, !containsURL = 0.28



