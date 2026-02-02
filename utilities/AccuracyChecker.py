import torch as t

def classifier_checker(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with t.no_grad():
        if loader.dataset.train:
            #print("Checking accuracy on training data")
            pass
        else:
            #print("Checking accuracy on test data")
            pass
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1) # dimension = (batch_size, num_classes)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy= float(num_correct)/float(num_samples)*100
        print(f"Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}")
    return accuracy