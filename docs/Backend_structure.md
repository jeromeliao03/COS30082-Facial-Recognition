# Backend Plans

I will first build the actual detector logic, this needs to exist for any of the models to actually run. 
This should be able to detect faces which will be passed to the actual models as input. 

Face recognition == 96x96 - 128-dim embdding
anti spoofing == 128x128 sigmoid 
emotion == 224x224 7-class softmax?

ALL TO BE 128