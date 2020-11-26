# trolley_bed_detection

### Data Collection
    1. Bing scapper
        - We scap around 600 images from [Bing](https://www.bing.com/?scope=images&nr=1&FORM=NOFORM). 200 images left after filtering those duplicate and irrelevant images.

    2. Manual collect
        - Nothing to describe. Collect around 200 image

### Training Phase
    

### Testing Phase
    1. Testing on Image
    ```bash
    python3 detect.py --source ../dataset/trolley_bed/test/images --weights ../runs/train/exp7_trolley_bed/weights/best.pt --conf 0.5
    ```

    2. Testing on Video
    ```bash
    python3 detect.py --source ../dataset/trolley_bed/0.mp4 --weights ../runs/train/exp7_trolley_bed/weights/best.pt --conf 0.5
    ```

    3. 



