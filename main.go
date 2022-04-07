package main

import (
	"fmt"
	"image"
	"image/color"
	"strconv"

	"gocv.io/x/gocv"
)

func main() {
	// 摄像头ID,RTSP地址均可
	deviceID := "rtsp://admin:admin@192.168.1.8:554/E:/2-video/aql.dav"
	model := "tf-model/frozen_inference_graph.pb"
	config := "tf-model/ssd_mobilenet_v1_coco_2017_11_17.pbtxt"
	// 捕获视频
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()
	// 创建windows窗口
	window := gocv.NewWindow("DNN Detection")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// 加载DNN神经网络模型
	net := gocv.ReadNet(model, config)
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, config)
		return
	}
	defer net.Close()
	err = net.SetPreferableBackend(gocv.NetBackendDefault)
	if err != nil {
		return
	}
	err = net.SetPreferableTarget(gocv.NetTargetCPU)
	if err != nil {
		return
	}

	var ratio = 1.0 / 127.5
	var mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)

	fmt.Printf("Start reading device: %v\n", deviceID)
	rgba := color.RGBA{G: 255}
	analyzes := make([]analyze, 0)
	times := 0
	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if !img.Empty() {
			if times%4 == 0 {
				// convert image Mat to 300x300 blob that the object detector can analyze
				blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, true, false)
				// feed the blob into the detector
				net.SetInput(blob, "")
				// run a forward pass thru the network
				prob := net.Forward("")
				analyzes = performDetection(&img, prob, rgba)
				prob.Close()
				blob.Close()
			} else {
				for _, ana := range analyzes {
					gocv.Rectangle(&img, image.Rect(ana.left, ana.top, ana.right, ana.bottom), rgba, 2)
					gocv.PutText(&img, ana.text, image.Pt(ana.left, ana.top), gocv.FontHersheyPlain, 1.5, rgba, 2)
				}
			}
			times++

			window.IMShow(img)
			if window.WaitKey(1) >= 0 {
				break
			}
		}
	}
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performDetection(frame *gocv.Mat, results gocv.Mat, rgba color.RGBA) []analyze {
	analyzes := make([]analyze, 0)
	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		if confidence > 0.6 {
			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), rgba, 2)
			float := strconv.FormatFloat(float64(results.GetFloatAt(0, i+1)), 'f', -1, 32)
			classId := int(results.GetFloatAt(0, i+1))
			if classId == 1 {
				float = "person"
			} else if classId == 3 {
				float = "car"
			} else if classId == 4 {
				float = "motor"
			}
			gocv.PutText(frame, float, image.Pt(left, top), gocv.FontHersheyPlain, 1.5, rgba, 2)

			analyzes = append(analyzes, analyze{classId, left, top, right, bottom, float})
		}
	}
	return analyzes
}

type analyze struct {
	classId, left, top, right, bottom int
	text                              string
}
