import torch
path = 'weights/best_model.pt'
state = torch.load(path, map_location='cpu')
if isinstance(state, dict) and 'state_dict' in state:
    sd = state['state_dict']
else:
    sd = state
print(type(sd))
print('num', len(sd))
print('keys:')
for k in list(sd.keys())[:40]:
    print(k)
print('\nselected shapes:')
for k, v in sd.items():
    if any(name in k for name in ['final', 'out_conv', 'decoder.up4', 'decoder.up1', 'encoder.conv1', 'encoder.conv2', 'encoder.conv3', 'encoder.conv4', 'bottleneck']):
        print(k, tuple(v.shape))
