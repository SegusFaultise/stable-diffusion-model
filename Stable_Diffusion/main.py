#region Imports
import torch
import os
from utils import *
from SDModules import UNet
import logging
from tqdm import tqdm
from rich import print
from SDUtils import getData, plotImages, setupLogging, saveImages
import flask
from flask import send_file
#endregion

#region Setting Up Flask Var {app}
app = flask.Flask(__name__)
#endregion

#region Logging
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")
#endregion

#region Diffusion Model
class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepareNoiseSchedule().to(device)
        self.aplha = 1. - self.beta
        self.aplha_hat = torch.cumprod(self.aplha, dim=0)

    def prepareNoiseSchedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noiseImages(self, x, t):
        sqrt_aplha_hat = torch.sqrt(self.aplha_hat[t])[:, None, None, None]
        sqrt_one_minus_aplha_hat = torch.sqrt(1 - self.aplha_hat[t])[:, None, None, None]
        e = torch.randn_like(x)
        return sqrt_aplha_hat * x + sqrt_one_minus_aplha_hat * e, e
    
    def sampleTimesSteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n, ))

    def sample(self, model, n):
        logging.info(F"Sampling {n} new images....")
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.aplha[t][:, None, None, None]
                alpha_hat = self.aplha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x
#endregion

#region Luanch trainning
def Launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_uncondistional"
    args.epohcs = 300
    args.batch_size = 7
    args.image_size = 64
    args.dataset_path = r"C:\Users\wilso\Downloads\Personal_Code_Projects\archive\train"
    args.device = "cuda"
    args.lr = 3e-4
#endregion

#region Get Request & Hosting Api
@app.route("/images", methods=["GET"])
def images():
        Launch()
        device = "cuda"
        model = UNet().to(device)
        ckpt = torch.load(r"C:\Users\wilso\Downloads\Personal_Code_Projects\STABLE_DIFFUSION_API\Saved_Model\ckpt.pt")
        model.load_state_dict(ckpt)
        diffusion = Diffusion(img_size=64, device=device)
        x = diffusion.sample(model, 10)
        saveImages(x, os.path.join("Api_Images", F"img.jpg"))
        file_name = r"C:\Users\wilso\Downloads\Personal_Code_Projects\STABLE_DIFFUSION_API\Api_Images\img.jpg"
        return send_file(file_name, mimetype="image/jpg")

if __name__ == "__main__":
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)
#endregion