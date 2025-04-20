import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocess.dataset import SessionDataset
from models.dual_transformer import DualStreamModel
from utils.config import load_config
from utils.logger import Logger
from tqdm import tqdm  # <-- import tqdm

def main(args):
    # Load configuration.
    config = load_config(args.config)
    seed = config.get("experiment", {}).get("seed", 42)
    torch.manual_seed(seed)
    
    # Setup log and checkpoint directories.
    log_dir = config["logging"]["log_dir"]
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = Logger(log_dir)
    
    # Initialize dataset and dataloader.
    data_dir = config["data"]["raw_data_dir"]
    segment_duration = config["data"].get("segment_duration", 10)
    mouse_sample_rate = config["data"].get("sample_rate_mouse", 100)
    max_mouse_len = config["data"].get("max_mouse_len", 1000)
    max_key_len = config["data"].get("max_key_len", 500)
    
    dataset = SessionDataset(data_dir, segment_duration, mouse_sample_rate, max_mouse_len, max_key_len)
    dataloader = DataLoader(dataset, batch_size=config["training"]["batch_size"],
                            shuffle=True, num_workers=4)
    
    # Model parameters.
    mouse_input_dim = 5     # [timestamp, x, y, vx, vy]
    keyboard_input_dim = 4  # [timestamp, key code, action, hold_time]
    embed_dim = config["model"]["embed_dim"]
    n_layers = config["model"]["n_layers"]
    n_heads = config["model"]["n_heads"]
    dropout = config["model"].get("dropout", 0.1)
    max_len = max(max_mouse_len, max_key_len)
    
    model = DualStreamModel(mouse_input_dim, keyboard_input_dim,
                            embed_dim, n_layers, n_heads, dropout, max_len)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Loss: Triplet loss.
    margin = config["training"].get("margin", 1.0)
    triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
    
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    n_epochs = config["training"]["n_epochs"]
    log_interval = config["training"].get("log_interval", 50)
    checkpoint_interval = config["training"].get("checkpoint_interval", 5)
    
    model.train()
    global_step = 0
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        # Wrap the dataloader with tqdm for a progress bar.
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
        for batch_idx, batch in progress_bar:
            x_mouse = batch["mouse"].to(device)       # [batch, seq_len, 5]
            x_keyboard = batch["keyboard"].to(device)   # [batch, seq_len, 4]
            # For demonstration, this example uses the first three samples as the triplet.
            batch_size = x_mouse.size(0)
            if batch_size < 3:
                continue
            
            embeddings = model(x_mouse, x_keyboard)  # shape: [batch, embed_dim]
            player_ids = batch["player_id"]
            # Ensure first two belong to the same player and third to a different one.
            if player_ids[0] != player_ids[1] or player_ids[0] == player_ids[2]:
                continue  # Skip batch if conditions are not met
            
            anchor = embeddings[0].unsqueeze(0)
            positive = embeddings[1].unsqueeze(0)
            negative = embeddings[2].unsqueeze(0)
            
            loss = triplet_loss_fn(anchor, positive, negative)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # Update progress bar with loss info.
            progress_bar.set_postfix(loss=loss.item())
            
            if global_step % log_interval == 0:
                logger.log("Epoch: {}, Step: {}, Loss: {:.4f}".format(epoch, global_step, loss.item()))
        
        avg_loss = epoch_loss / (batch_idx + 1)
        print("Epoch {} Average Loss: {:.4f}".format(epoch, avg_loss))
        logger.log("Epoch {} Average Loss: {:.4f}".format(epoch, avg_loss))
        
        if epoch % checkpoint_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, ckpt_path)
            print("Saved checkpoint to", ckpt_path)
            logger.log("Saved checkpoint to {}".format(ckpt_path))
    
    print("Training completed.")
    logger.log("Training completed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Dual-Stream Transformer for Behavioral Authentication")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
