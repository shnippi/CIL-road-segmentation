import os
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

def get_test_fn(config):
    return test_fn


def test_fn(models, test_dataloader, config, device):
    gen = models['gen']
    result_path = config['result_dir'] + "/" + config['model'] + "_" \
        + config['generation_mode']  + "_" + config['transformation']
    
    if config['transformation'] == 'label':
        result_path = result_path + "/masks"
    else:
        result_path = result_path + "/roadmaps"

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    with tqdm(test_dataloader) as tepoch:
        count = 144
        trans_tensToPil = transforms.ToPILImage()
        for batch, data in enumerate(tepoch):
            # X:= Sattelite, Y:= Roadmap
            A = data['A']
            A = A.to(device)

            B_fake = gen(A)
            
            pil_img = trans_tensToPil(B_fake.squeeze())
            pil_img = pil_img.resize((400, 400))

            path = result_path + "/" + str(count) + ".png"
            pil_img.save(path)
            count += 1
