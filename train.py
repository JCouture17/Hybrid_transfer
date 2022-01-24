### Train model ###
import numpy as np
import torch.nn as nn
import torch
import copy
from time import time
from torch.optim import Adam, lr_scheduler


class train:
    
    def mape(y_true, y_pred): 
        mape = (torch.sum(torch.div(torch.abs(torch.sub(y_true, y_pred)), torch.abs(y_true))))*100
        return mape     
    
    def train_hybrid(model, model_name, train_his, test_his, train_images, test_images, lr, epochs, early_stop=5, opt='Adam'):
        t0 = time()
        early_stopping = EarlyStopping(patience=early_stop)
        optimizer = Adam(model.parameters(), lr=lr)
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # Loss Criterion
        criterion = nn.MSELoss()
        train_loss = []
        val_loss = []
        best_train, best_val = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            t1 = time()
            # Train and Validate
            print('epoch:', epoch)
            train_stats = train.train_hybrid_step(model, criterion, optimizer, train_his, train_images)
            valid_stats = train.valid_hybrid_step(model, criterion, test_his, test_images)
            train_loss.append(train_stats['loss'])
            val_loss.append(valid_stats['loss'])
            # Keep best model
            if valid_stats['accuracy'] > best_val or (valid_stats['MAE']==best_val and train_stats['accuracy']>=best_train):
                best_train  = train_stats['accuracy']
                print('training accuracy =', float(train_stats['accuracy']), '%')
                best_val    = valid_stats['accuracy']
                print('validation accuracy = ', float(valid_stats['accuracy']), '%')
                print('RMSE = ', float(valid_stats['RMSE']))
                print('MAE = ', float(valid_stats['MAE']))
                best_model_weights = copy.deepcopy(model.state_dict())
            if epoch % 5 == 0: # Update the learning rate every 5 epochs
                lr_decay.step()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Time taken for epoch = %fs' % (time() - t1))
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_weights)
        test_stats = train.valid_hybrid_step(model, criterion, test_his, test_images)
        print("Total time = %ds" % (time() - t0))
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                    test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
        save_dir = "./result"
        torch.save(model.state_dict(), save_dir + '/trained_hybrid_' + model_name + '.pkl') # Use this to save the model to a .pkl file
        # print('Trained model saved to \'%s/trained_hybrid_' + model_name + '.pkl\'' % save_dir)
        return model, train_loss, val_loss
    
    def train_lstm(train_loader, test_loader, lr, epochs, model, early_stop=5, opt='Adam'):
        t0 = time()
        early_stopping = EarlyStopping(patience=early_stop)
        optimizer = Adam(model.parameters(), lr=lr)
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        # Loss Criterion
        criterion = nn.MSELoss()
        train_loss = []
        val_loss = []
        best_train, best_val = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            t1 = time()
            # Train and Validate
            print('epoch:', epoch)
            train_stats = train.train_lstm_step(model, criterion, optimizer, train_loader)
            valid_stats = train.valid_lstm_step(model, criterion, test_loader)
            train_loss.append(train_stats['loss'])
            val_loss.append(valid_stats['loss'])
            # Keep best model
            if valid_stats['accuracy'] > best_val or (valid_stats['MAE']==best_val and train_stats['accuracy']>=best_train):
                best_train  = train_stats['accuracy']
                print('training accuracy =', float(train_stats['accuracy']), '%')
                best_val    = valid_stats['accuracy']
                print('validation accuracy = ', float(valid_stats['accuracy']), '%')
                print('RMSE = ', float(valid_stats['RMSE']))
                print('MAE = ', float(valid_stats['MAE']))
                best_model_weights = copy.deepcopy(model.state_dict())
            if epoch % 5 == 0: # Decay the learning rate every 5 epochs
                lr_decay.step()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Time taken for epoch = %fs' % (time() - t1))
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_weights)
        test_stats = train.valid_lstm_step(model, criterion, test_loader)
        print("Total time = %ds" % (time() - t0))
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                    test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
        save_dir = "./result"
        torch.save(model.state_dict(), save_dir + '/trained_lstm.pkl') # Use this to save the model to a .pkl file
        # print('Trained model saved to \'%s/trained_lstm.pkl\'' % save_dir)
        return model, train_loss, val_loss    
        
    def train_transfer_network(model, training, validation, lr, epochs, model_name, early_stop=5, opt='Adam'):
        t0 = time()
        model = model
        early_stopping = EarlyStopping(patience=early_stop)
        optimizer = Adam(model.parameters(), lr=lr)
        lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        train_loader, test_loader = training, validation      
        # Loss Criterion
        criterion = nn.MSELoss()
        train_loss = []
        val_loss = []
        best_train, best_val = 0.0, 0.0
        for epoch in range(1, epochs + 1):
            t1 = time()
            # Train and Validate
            print('epoch:', epoch)
            train_stats = train.train_step(model, criterion, optimizer, train_loader, epoch)
            valid_stats = train.valid_step(model, criterion, test_loader)
            train_loss.append(train_stats['loss'])
            val_loss.append(valid_stats['loss'])
            # Keep best model
            if valid_stats['accuracy'] > best_val or (valid_stats['MAE']==best_val and train_stats['accuracy']>=best_train):
                best_train  = train_stats['accuracy']
                print('training accuracy =', float(train_stats['accuracy']), '%')
                best_val    = valid_stats['accuracy']
                print('validation accuracy = ', float(valid_stats['accuracy']), '%')
                print('RMSE = ', float(valid_stats['RMSE']))
                print('MAE = ', float(valid_stats['MAE']))
                best_model_weights = copy.deepcopy(model.state_dict())
            if epoch % 5 == 0:    
                lr_decay.step()
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            print('Time taken for epoch = %fs' % (time() - t1))
        # Load best model and evaluate on test set
        model.load_state_dict(best_model_weights)
        test_stats = train.valid_step(model, criterion, test_loader)
        print("Total time = %ds" % (time() - t0))
        print('\nBest Validation Results: Average Loss: {:4.2f} | Accuracy: {:4.2f} | MAE: {:4.2f} | RMSE: {:4.2f}'.format(test_stats['loss'],
                                                                    test_stats['accuracy'], test_stats['MAE'], test_stats['RMSE']))
        save_dir = "./result"
        torch.save(model.state_dict(), save_dir + '/trained_transfer_' + model_name +'.pkl') # Use this to save the model to a .pkl file
        # print('Trained model saved to \'%s/trained_transfer_' + model_name + '.pkl\'' % save_dir)
        return model, train_loss, val_loss
        
    def valid_step(model, criterion, val_loader):
        model.eval()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        mse, mae, se = 0.0, 0.0, 0.0
        for i, (x_imgs, labels) in enumerate(val_loader):
            x_imgs, labels = x_imgs.cuda(), labels.cuda()
            # forward pass
            output = model(x_imgs)
            loss = criterion(output, labels)
            # gather statistics
            avg_loss += loss.item()        
            mae += torch.sum((abs(output - labels)))
            # print('outputs ', output)
            # print('labels ', labels)
            # print('mae ', torch.sum((abs(output - labels))))
            se += torch.sum((abs(output - labels))**2)
            # print('se ', torch.sum((abs(output - labels))**2))
            absolute_percentage_error += train.mape(labels, output)
            # print('ape ', train.mape(labels, output))
            # print('mse loss ', loss.item())
            mse += nn.MSELoss(output, labels, reduction='sum')
        rmse = (mse / len(val_loader.dataset))**0.5
        return {'loss' : avg_loss / len(val_loader.dataset), 'accuracy' : 100-(absolute_percentage_error / len(val_loader.dataset)),
                'MAE' : mae / len(val_loader.dataset), 'RMSE' : rmse}
    
    def train_step(model, criterion, optimizer, train_loader, epoch):
        model.train()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        for i, (x_imgs, labels) in enumerate(train_loader):
            x_imgs, labels = x_imgs.cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward pass
            pred = model(x_imgs)
            loss = criterion(pred, labels.float())
            loss.backward()
            optimizer.step()
            # gather statistics
            avg_loss += loss.item()
            absolute_percentage_error += train.mape(labels, pred)
        # writer.add_scalar("Loss/train", avg_loss / len(train_loader.dataset), epoch)
        return {'loss': avg_loss / len(train_loader.dataset), 'accuracy': 100-(absolute_percentage_error / len(train_loader.dataset))}
    
    def train_lstm_step(model, criterion, optimizer, train_loader):
        model.train()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        for i, (x_his, labels) in enumerate(train_loader):
            x_his, labels = x_his.float().cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward pass
            pred = model(x_his)
            # pred = torch.squeeze(pred, 2)
            loss = criterion(pred, labels.float())
            loss.backward()
            optimizer.step()
            # gather statistics
            avg_loss += loss.item()
            # print('labels ', labels)
            # print('predictions ', pred)            
            absolute_percentage_error += train.mape(labels, pred)
            # print('absolute error ', absolute_percentage_error)
        return {'loss': avg_loss / len(train_loader.dataset), 'accuracy': 100-(absolute_percentage_error / len(train_loader.dataset))}    
    
    def valid_lstm_step(model, criterion, val_loader):
        model.eval()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        mae, se = 0.0, 0.0
        for i, (x_his, labels) in enumerate(val_loader):
            x_his, labels = x_his.float().cuda(), labels.cuda()
            # forward pass
            output = model(x_his)
            # output = torch.squeeze(output, 2)
            loss = criterion(output, labels)
            # gather statistics
            avg_loss += loss.item()        
            mae += torch.sum((abs(torch.sub(output, labels))))
            se += torch.sum((abs(output - labels))**2)
            absolute_percentage_error += train.mape(labels, output)
        rmse = ((se/(len(val_loader.dataset)))**0.5)
        # print('Validation accuracy: ', 100-(absolute_percentage_error / len(val_loader.dataset)))
        return {'loss' : avg_loss / len(val_loader.dataset), 'accuracy' : 100-(absolute_percentage_error / len(val_loader.dataset)),
                'MAE' : mae / len(val_loader.dataset), 'RMSE' : rmse}
    
    def train_hybrid_step(model, criterion, optimizer, train_his, train_images):
        model.train()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        dataloader_iter = iter(train_images) # Image dataset
        for i, (x_his, labels) in enumerate(train_his):
            # print('Health indicator labels: ', labels)
            try:
                (x_images, labels) = next(dataloader_iter)
            except StopIteration:
                print("Something doesn't work")
            # print('Images labels: ', labels)
            x_his, x_images, labels = x_his.float().cuda(), x_images.cuda(), labels.cuda()
            optimizer.zero_grad()
            # forward pass
            pred = model(x_his, x_images)
            loss = criterion(pred, labels.float())
            loss.backward()
            optimizer.step()
            # gather statistics
            avg_loss += loss.item()
            absolute_percentage_error += train.mape(labels, pred)
        return {'loss': avg_loss / len(train_his.dataset), 'accuracy': 100-(absolute_percentage_error / len(train_his.dataset))}
        
    def valid_hybrid_step(model, criterion, test_his, test_images):
        model.eval()
        avg_loss = 0.0
        absolute_percentage_error = 0.0
        mae, se = 0.0, 0.0
        dataloader_iter = iter(test_images)
        for i, (x_his, labels) in enumerate(test_his):
            try:
                (x_images, labels) = next(dataloader_iter)
            except StopIteration:
                print("Something doesn't work")
            x_his, x_images, labels = x_his.float().cuda(), x_images.cuda(), labels.cuda()
            # forward pass
            output = model(x_his, x_images)
            loss = criterion(output, labels)
            # gather statistics
            avg_loss += loss.item()        
            mae += torch.sum((abs(torch.sub(output, labels))))
            se += torch.sum((abs(output - labels))**2)
            absolute_percentage_error += train.mape(labels, output)
        rmse = ((se/(len(test_his.dataset)))**0.5)
        return {'loss' : avg_loss / len(test_his.dataset), 'accuracy' : 100-(absolute_percentage_error / len(test_his.dataset)),
                'MAE' : mae / len(test_his.dataset), 'RMSE' : rmse}
        
    
    def test_transfer(model, test_loader):
        criterion = nn.MSELoss()
        test_stats = train.valid_step(model, criterion, test_loader)
        return test_stats
    
    def test_lstm(model, test_loader):
        criterion = nn.MSELoss()
        test_stats = train.valid_lstm_step(model, criterion, test_loader)
        return test_stats
    
    def test_hybrid(model, test_his, test_images):
        criterion = nn.MSELoss()
        test_stats = train.valid_hybrid_step(model, criterion, test_his, test_images)
        return test_stats