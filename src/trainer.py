import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.dataset import DigitalSoulDataset
from src.model import DigitalSoulModel
from src.utils import save_checkpoint
import config

def train_model():
    print("🚀 Starting Training Pipeline...")
    
    # 1. Prepare Data
    full_dataset = DigitalSoulDataset()
    
    # Simple Train/Val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_data, val_data = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config.BATCH_SIZE)
    
    # 2. Init Model
    model = DigitalSoulModel().to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    criterion = nn.L1Loss() # Mean Absolute Error (MAE)
    
    # --- TRACKING VARIABLES ---
    best_val_loss = float('inf')
    patience_counter = 0
    PATIENCE_LIMIT = 15  # Stop if no improvement for 15 epochs

    # 3. Loop
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        
        for ling, acou, labels in train_loader:
            ling, acou, labels = ling.to(config.DEVICE), acou.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(ling, acou)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for ling, acou, labels in val_loader:
                ling, acou, labels = ling.to(config.DEVICE), acou.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(ling, acou)
                total_val_loss += criterion(outputs, labels).item()
        
        avg_val_loss = total_val_loss / len(val_loader)

        # Update Scheduler
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{config.EPOCHS}] - Train MAE: {avg_train_loss:.4f} | Val MAE: {avg_val_loss:.4f}")
        
        # --- SAVE BEST MODEL ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0 # Reset counter
            print(f"   🔥 New Best Model! (MAE: {best_val_loss:.4f}) -> Saving...")
            save_checkpoint(model, optimizer, filename="digital_soul_final.pth")
        else:
            patience_counter += 1
            
        # --- EARLY STOPPING ---
        if patience_counter >= PATIENCE_LIMIT:
            print(f"\n🛑 Early Stopping triggered! No improvement for {PATIENCE_LIMIT} epochs.")
            print(f"   Best Validation MAE was: {best_val_loss:.4f}")
            break

    print("🏁 Training Complete!")