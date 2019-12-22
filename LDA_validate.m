function succ_perc = LDA_validate(W, b, validation_trials, validation_labels)
 %LDA_validate Compute classification success rate on given trials, using
 %linear weights
 % W,b - weights&bias to classify with
 % validation_trials, validation_labels - trials to classify, and the'r
 %      correct labels for comparison
 
    val_classification = sign(validation_trials*W+b);
    val_success = sum(val_classification == validation_labels);
    succ_perc = val_success/size(validation_trials,1);
end

