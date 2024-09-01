from src.app.data_extraction.training_history import AccuracyHistory, LossHistory

acc_history = AccuracyHistory(fig_name='test_accuracy_history')
loss_history = LossHistory(fig_name='test_loss_history')


mock_acc_data = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
mock_loss_data = [(1.0, 0.8), (0.7, 0.6), (0.4, 0.3), (0.2, 0.1)]

for data in mock_acc_data:
    acc_history.add(data)
    
for i, data in enumerate(mock_loss_data):
    loss_history.add(*data)
    
acc_history.render()
loss_history.render()