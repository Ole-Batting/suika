import cv2
import matplotlib.pyplot as plt
import numpy as np
from icecream import ic


class Tracker:
    def __init__(self):
        self.initialized = False
        self.clicked_x = None
        self.clicked_y = None
        self.graph_list = []
        self.pt_list = []
        self.m_list = []
        self.radius = 16
        self.frame_num = 0
        self.frame_last = None
        self.min = None
        self.fps = None
        cv2.namedWindow("tracker")
        cv2.setMouseCallback('tracker', self.mouse_click_handler)

    def mouse_click_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            ic(x, y)
            self.clicked_x = x
            self.clicked_y = y
            self.pt_list.append([self.frame_num, x, y])
            self.initialized = True

    @property
    def prev_pt(self):
        return self.pt_list[-1]

    @property
    def pts(self):
        if len(self.pt_list) != 0:
            self.next_graph()
        return self.graph_list

    def set_frame_num(self, val):
        self.frame_num = val

    def next_graph(self):
        arr = np.array(self.pt_list)
        arr[:, 2] -= self.min

        self.graph_list.append(arr)

        t = arr[:, 0] / self.fps
        y = np.sqrt(arr[:, 2])
        n = len(t)

        fig = plt.figure()
        plt.plot(t, y)
        plt.show()

        s_x = np.sum(t)
        s_xx = np.sum(np.square(t))
        s_y = np.sum(y)
        s_yy = np.sum(np.square(y))
        s_xy = np.sum(t * y)

        m = ((n * s_xy) - (s_x * s_y)) / ((n * s_xx) - (s_x * s_x))

        self.m_list.append(m)

        self.pt_list = []

    def set_min(self):
        self.min = self.pt_list[0][2]
        ic(self.min)
        self.pt_list = []

    def set_fps(self, fps):
        self.fps = fps
        ic(self.fps)


if __name__ == "__main__":
    #cap = cv2.VideoCapture("/Users/olebatting/Downloads/capture (3) (online-video-cutter.com).mp4")
    cap = cv2.VideoCapture("/Users/olebatting/Desktop/ml-repos/suika/scenes/playback.mp4")
    ret, frame = cap.read()
    last_frame = frame[:-100]
    trk = Tracker()
    trk.set_fps(cap.get(cv2.CAP_PROP_FPS))
    wait = 0

    while ret:
        ret, raw = cap.read()
        frame = raw[:-100]

        cv2.imshow("tracker", raw)

        print(np.sum(frame - last_frame))
        #if np.sum(frame - last_frame) < 8_000_000:
        if np.sum(frame - last_frame) < 1_000_000:
            key = cv2.waitKey(10)
        else:
            key = cv2.waitKey(10 * wait)
        if key == ord("q"):
            break
        elif key == ord("p"):
            wait = 1 - wait
        elif key == ord("n"):
            trk.next_graph()
        elif key == ord("s"):
            trk.set_min()

        trk.set_frame_num(cap.get(cv2.CAP_PROP_POS_FRAMES))
        last_frame = frame.copy()

    cap.release()
    cv2.destroyAllWindows()

    pts = trk.pts

    print(trk.m_list)
    fig, ax = plt.subplots(1, 2)
    for i in range(len(pts)):
        arr = pts[i]
        ax[0].plot(arr[:, 0], arr[:, 2])
        acc = arr.copy()
        acc[:, 2] **= 0.5
        acc[:, 2] /= acc[:, 0] + 1e-5
        ax[1].plot(acc[:, 0], acc[:, 2])
    plt.show()
