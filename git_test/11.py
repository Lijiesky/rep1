from tensorboard.backend.event_processing import event_accumulator

#加载日志数据
ea=event_accumulator.EventAccumulator(r'D:\pycharmproject\Tensorboard_test\logs\scalars\20210705-104429\validation\events.out.tfevents.1625453075.DESKTOP-NM9CGBU.792.493.v2')
ea.Reload()
print(ea.scalars.Keys())

val_acc=ea.scalars.Items('evaluation_loss_vs_iterations')
print(len(val_acc))
print([(i.step,i.value) for i in val_acc])
