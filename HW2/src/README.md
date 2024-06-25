# Camera Relocalization

### Requirements

* Python == 3.8
* open3d == 0.13.0
* pandas == 1.4.1
* imageio == 2.16.1
* scipy == 1.8.0
* opencv-python == 4.5.5.64 
* numpy == 1.22.2

### Dataset

To run the code, please run following to get dataset

```
cd data
bash get_dataset.sh
cd ..
```

## Usage

### Part 1-1 and Part 1-2

#### Using P3P or DLT solve PnP problem

```
python3 p1_12.py --method P3P
python3 p1_12.py --method DLT
```

### Part 1-3

```
python3 p1_3.py
```

### Part 2

```
python3 p2.py
```

## AR cube video

Draw cube on NTU gate

![Alt Text](./output/cube.gif)
