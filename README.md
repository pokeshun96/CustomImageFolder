# CustomImageFolder
pytorchで画像分類タスクを行うときに、不正解データの画像ファイルを出力したい！<br>
ImageFolder クラスを使っていて、ファイルパスも返すように DataLoader を調整するために、カスタムの Dataset クラスを作成する.

```
from torchvision.datasets import ImageFolder

class CustomImageFolder(ImageFolder):
    def __getitem__(self, index):
        # 元の ImageFolder クラスの __getitem__ メソッドを呼び出す
        original_tuple = super(CustomImageFolder, self).__getitem__(index)
        # ファイルパスを取得
        path = self.imgs[index][0]
        # 画像データ、ラベル、ファイルパスを含むタプルを返す
        return (original_tuple + (path,))
```

そして、この CustomImageFolder クラスを使用してデータセットを作成します. 

```
# カスタムデータセットのインスタンスを作成
image_datasets = {
    x: CustomImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

# DataLoader の作成
dataloaders = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                   shuffle=True, num_workers=4, pin_memory=True)
    for x in ['train', 'val']
}
```

これで、DataLoader から得られる各バッチは (inputs, labels, paths) の形式になります。<br>
これを使って、train_model 関数内で不正解の画像のファイルパスを取得して出力することができます.

```
# 検証フェーズ
if phase == 'val':
    # ...
    for inputs, labels, paths in dataloaders['val']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # ...
        incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]
        for idx in incorrect_indices:
            incorrect_samples.append(paths[idx])
    # ...

# 検証終了後に不正解データのファイル名の出力
print("Incorrect samples:")
for file_path in incorrect_samples:
    print(file_path)
```
