import torch as t

def save_checkpoint(model, optimizer, filename):
    checkpoint = {'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    print("=> Saving checkpoint")
    t.save(checkpoint, filename)

def load_checkpoint(model, optimizer, checkpoint):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
