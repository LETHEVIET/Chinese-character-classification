
@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    frac = 0
    deno = 0

    for (images, labels) in chinese_val_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()

        _, preds = outputs.max(1)
        label_names = [idx_to_class[label_id.item()] for label_id in labels]
        pred_names  = [idx_to_class[pred_id.item()] for pred_id in preds]

        frac += sum([editdistance.eval(pred_names[i], label_names[i]) for i in range(len(label_names))])
        deno += sum([1 if label_names[i] == "zc" else len(label_names[i]) for i in range(len(label_names))])

        correct += preds.eq(labels).sum()

    levenshtein_accuracy = 1 - float(frac) / deno

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Levenshtein accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(chinese_val_loader.dataset),
        correct.float() / len(chinese_val_loader.dataset),
        levenshtein_accuracy,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(chinese_val_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(chinese_val_loader.dataset), epoch)
        writer.add_scalar('Test/Levenshtein accuracy', levenshtein_accuracy, epoch)

    return correct.float() / len(chinese_val_loader.dataset)