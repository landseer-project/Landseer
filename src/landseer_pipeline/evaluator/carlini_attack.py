import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CarliniL2Attack:
    """
    Carlini & Wagner L2 Attack implementation for MagNet evaluation
    Based on the original paper: "Towards Evaluating the Robustness of Neural Networks"
    """
    def __init__(self, model, device, confidence=0, learning_rate=0.01, 
                 max_iterations=1000, binary_search_steps=5, initial_const=1e-3):
        self.model = model
        self.device = device
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        
    def __call__(self, images, labels):
        """
        Generate adversarial examples using C&W L2 attack
        Args:
            images: Input images
            labels: Integer labels (not one-hot)
        """
        batch_size = images.shape[0]
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Convert to [-1, 1] range for optimization
        boxmin, boxmax = 0, 1
        
        # Initialize variables
        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)
        const = torch.full((batch_size,), self.initial_const, device=self.device)
        
        best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        best_adv = images.clone()
        
        # Binary search
        for search_step in range(self.binary_search_steps):
            # Initialize perturbation variable
            delta = torch.zeros_like(images, requires_grad=True)
            optimizer = optim.Adam([delta], lr=self.learning_rate)
            
            for iteration in range(self.max_iterations):
                optimizer.zero_grad()
                
                # Apply perturbation and clamp to valid range
                adv_images = torch.clamp(images + delta, boxmin, boxmax)
                
                # Forward pass
                outputs = self.model(adv_images)
                
                # Calculate losses
                l2_loss = torch.sum((adv_images - images) ** 2, dim=[1, 2, 3])
                
                # Carlini objective function - work with integer labels
                # Extract scores for true classes using gather
                real = torch.gather(outputs, 1, labels.unsqueeze(1)).squeeze(1)
                
                # Find max score among non-true classes
                outputs_masked = outputs.clone()
                outputs_masked.scatter_(1, labels.unsqueeze(1), -1e4)  # Mask true class
                other = torch.max(outputs_masked, dim=1)[0]
                
                loss1 = torch.clamp(real - other + self.confidence, min=0)
                
                # Total loss
                loss = torch.sum(const * loss1 + l2_loss)
                
                loss.backward()
                optimizer.step()
                
                # Update best adversarial examples
                pred_labels = torch.argmax(outputs, dim=1)
                for i in range(batch_size):
                    if l2_loss[i] < best_l2[i] and pred_labels[i] != labels[i]:
                        best_l2[i] = l2_loss[i]
                        best_adv[i] = adv_images[i]
            
            # Update binary search bounds
            with torch.no_grad():
                final_pred = torch.argmax(self.model(best_adv), dim=1)
                for i in range(batch_size):
                    if final_pred[i] != labels[i]:
                        upper_bound[i] = const[i]
                    else:
                        lower_bound[i] = const[i]
                
                if upper_bound[i] < 1e9:
                    const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    const[i] = lower_bound[i] * 10
        
        return best_adv

def evaluate_carlini_l2(model, loader, device, confidence=0):
    """
    Evaluate model against Carlini L2 attack
    """
    model.eval()
    attack = CarliniL2Attack(model, device, confidence=confidence)
    
    correct, total = 0, 0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        
        # Use integer labels directly - no one-hot conversion needed
        adv_X = attack(X, y)
        
        with torch.no_grad():
            pred = model(adv_X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    
    return correct / total if total > 0 else 0.0
