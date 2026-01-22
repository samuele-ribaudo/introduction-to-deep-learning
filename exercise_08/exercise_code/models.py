import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

class Encoder(nn.Module):

    def __init__(self, hparams, input_size=28 * 28, latent_dim=20):
        super().__init__()

        # set hyperparams
        self.latent_dim = latent_dim 
        self.input_size = input_size
        self.hparams = hparams
        self.encoder = None

        ########################################################################
        # TODO: Initialize your encoder!                                       #                                       
        #                                                                      #
        # Possible layers: nn.Linear(), nn.BatchNorm1d(), nn.ReLU(),           #
        # nn.Sigmoid(), nn.Tanh(), nn.LeakyReLU().                             # 
        # Look online for the APIs.                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Wrap them up in nn.Sequential().                                     #
        # Example: nn.Sequential(nn.Linear(10, 20), nn.ReLU())                 #
        #                                                                      #
        # Hint 2:                                                              #
        # The latent_dim should be the output size of your encoder.            # 
        # We will have a closer look at this parameter later in the exercise.  #
        ########################################################################

        n_hidden = self.hparams["n_hidden"]
        self.latent_dim = self.hparams["latent_dim"] if "latent_dim" in self.hparams else latent_dim

        """
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, self.latent_dim)
        )

        """
        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_hidden//2),
            nn.LeakyReLU(),
            nn.Linear(n_hidden//2, self.latent_dim)
        ) 
        
        

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into encoder!
        return self.encoder(x)

class Decoder(nn.Module):

    def __init__(self, hparams, latent_dim=20, output_size=28 * 28):
        super().__init__()

        # set hyperparams
        self.hparams = hparams
        self.decoder = None

        ########################################################################
        # TODO: Initialize your decoder!                                       #
        ########################################################################

        self.latent_dim = latent_dim
        self.output_size = output_size
        n_hidden = self.hparams["n_hidden"]
        self.latent_dim = self.hparams["latent_dim"] if "latent_dim" in self.hparams else latent_dim
        
        """
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, output_size),
            nn.Sigmoid()
        )

        """
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, n_hidden//2),
            nn.LeakyReLU(),
            nn.Linear(n_hidden//2, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, output_size),
            nn.Sigmoid()
        )
        
        

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        # feed x into decoder!
        return self.decoder(x)


class Autoencoder(nn.Module):

    def __init__(self, hparams, encoder, decoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        # Define models
        self.encoder = encoder
        self.decoder = decoder
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.set_optimizer()

    def forward(self, x):
        reconstruction = None
        ########################################################################
        # TODO: Feed the input image to your encoder to generate the latent    #
        #  vector. Then decode the latent vector and get your reconstruction   #
        #  of the input.                                                       #
        ########################################################################
        
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return reconstruction

    def set_optimizer(self):

        self.optimizer = None
        ########################################################################
        # TODO: Define your optimizer.                                         #
        ########################################################################
        
        self.optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"]
        )
        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def training_step(self, batch, loss_func):
        """
        This function is called for every batch of data during training. 
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the training step, similarly to the way it is shown in      #
        # train_classifier() in the notebook, following the deep learning      #
        # pipeline.                                                            #
        #                                                                      #
        # Hint 1:                                                              #
        # Don't forget to reset the gradients before each training step!       #
        #                                                                      #
        # Hint 2:                                                              #
        # Don't forget to set the model to training mode before training!      #
        #                                                                      #
        # Hint 3:                                                              #
        # Don't forget to reshape the input, so it fits fully connected layers.#
        #                                                                      #
        # Hint 4:                                                              #
        # Don't forget to move the data to the correct device!                 #                                     
        ########################################################################

        self.train()  # Set the model to training mode

        self.optimizer.zero_grad() # Reset the gradients - VERY important! Otherwise they accumulate.

        images = batch.to(self.device) # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.

        images = images.view(images.size(0), -1)  # Reshape the images to fit fully connected layers.


        reconstruction = self(images)
        loss = loss_func(reconstruction, images)
        
        loss.backward()  # Stage 2: Backward().
        self.optimizer.step() # Stage 3: Update the parameters.

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def validation_step(self, batch, loss_func):
        """
        This function is called for every batch of data during validation.
        It should return the loss for the batch.
        """
        loss = None
        ########################################################################
        # TODO:                                                                #
        # Complete the validation step, similraly to the way it is shown in    #
        # train_classifier() in the notebook.                                  #
        #                                                                      #
        # Hint 1:                                                              #
        # Here we don't supply as many tips. Make sure you follow the pipeline #
        # from the notebook.                                                   #
        ########################################################################
        
        self.eval()  # Set the model to evaluation mode
        images = batch.to(self.device) # Get the images and labels from the batch, in the fashion we defined in the dataset and dataloader.

        images = images.view(images.size(0), -1)
       
        with torch.no_grad():
            reconstruction = self(images)
            loss = loss_func(reconstruction, images)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return loss

    def getReconstructions(self, loader=None):

        assert loader is not None, "Please provide a dataloader for reconstruction"
        self.eval()
        self = self.to(self.device)

        reconstructions = []

        for batch in loader:
            X = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            reconstruction = self.forward(flattened_X)
            reconstructions.append(
                reconstruction.view(-1, 28, 28).cpu().detach().numpy())

        return np.concatenate(reconstructions, axis=0)


class Classifier(nn.Module):

    def __init__(self, hparams, encoder):
        super().__init__()
        # set hyperparams
        self.hparams = hparams
        self.encoder = encoder
        self.model = nn.Identity()
        self.device = hparams.get("device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        
        ########################################################################
        # TODO:                                                                #
        # Given an Encoder, finalize your classifier, by adding a classifier   #   
        # block of fully connected layers.                                     #                                                             
        ########################################################################

        n_hidden = self.hparams["n_hidden"]
        n_classes = self.hparams["n_classes"]
        self.latent_dim = self.hparams["latent_dim"] if "latent_dim" in self.hparams else 20

        for p in self.encoder.parameters():
            p.requires_grad = False

        """
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, n_hidden//4),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden//4, n_classes)
        )
        """

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, n_hidden//2),
            nn.BatchNorm1d(n_hidden//2),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(n_hidden//2, n_hidden//4),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(n_hidden//4, n_classes)
        )

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.set_optimizer()
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.model(x)
        return x

    def set_optimizer(self):
        
        self.optimizer = None
        ########################################################################
        # TODO: Implement your optimizer. Send it to the classifier parameters #
        # and the relevant learning rate (from self.hparams)                   #
        ########################################################################

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"],
        )

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def getAcc(self, loader=None):
        
        assert loader is not None, "Please provide a dataloader for accuracy evaluation"

        self.eval()
        self = self.to(self.device)
            
        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            flattened_X = X.view(X.shape[0], -1)
            score = self.forward(flattened_X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        preds = scores.argmax(axis=1)
        acc = (labels == preds).mean()
        return preds, acc
