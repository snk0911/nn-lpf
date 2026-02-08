import torch
import torchvision
# ... weitere Importe ...

# WICHTIG: Deine Klasse heißt wahrscheinlich 'ResNet' im Code, nicht 'MyResNet18'
from .resnet import resnet18 
# oder je nachdem, wie du die Funktion resnet18() in resnet.py aufrufst: 
# from resnet import resnet18 


# Lade das originale torchvision-Modell
original_model = torchvision.models.resnet18(pretrained=False, num_classes=200)

# Instanziiere DEIN Modell als BASELINE (aa_type='none')
# WICHTIG: Hier nimmst du deine Hauptklasse oder Hilfsfunktion:
my_model = resnet18(aa_type='none')
# ODER (falls du eine Helferfunktion resnet18() hast, die diese Parameter entgegennimmt):
# my_model = resnet18(aa_type='none')


# Gewichte übertragen
# Das sollte jetzt funktionieren, da die Strukturen identisch sind
my_model.load_state_dict(original_model.state_dict())

# Test mit einem zufälligen Bild
input_tensor = torch.randn(1, 3, 224, 224)
output_orig = original_model(input_tensor)
output_mine = my_model(input_tensor)

is_identical = torch.allclose(output_orig, output_mine, atol=1e-6)
print(f"original baseline and own baseline implementation within unified arch is same: {is_identical}")
