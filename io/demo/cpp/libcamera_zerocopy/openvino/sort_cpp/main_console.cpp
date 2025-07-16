#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include "Sort.h"
#include <iostream>
#include <vector>

// Constants
const float CONFIDENCE_THRESHOLD = 0.25;
const float NMS_THRESHOLD = 0.45;
const int IMG_WIDTH = 640;
const int IMG_HEIGHT = 640;

// Pre-processing function
cv::Mat preprocess(const cv::Mat& frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1. / 255., cv::Size(IMG_WIDTH, IMG_HEIGHT), cv::Scalar(), true, false);
    return blob;
}

// Post-processing function
std::vector<TrackingBox> postprocess(const ov::Tensor& output_tensor, const cv::Size& original_shape, Sort& tracker)
{
    const float* detections = output_tensor.get_data<const float>();
    std::vector<cv::Rect> boxes;
    std::vector<float> confidences;
    std::vector<int> class_ids;

    const ov::Shape output_shape = output_tensor.get_shape();
    const int rows = output_shape[1];
    const int cols = output_shape[2];

    for (int i = 0; i < rows; ++i)
    {
        float confidence = detections[i * cols + 4];
        if (confidence > CONFIDENCE_THRESHOLD)
        {
            float x = detections[i * cols + 0];
            float y = detections[i * cols + 1];
            float w = detections[i * cols + 2];
            float h = detections[i * cols + 3];

            int left = static_cast<int>((x - w / 2) * original_shape.width / IMG_WIDTH);
            int top = static_cast<int>((y - h / 2) * original_shape.height / IMG_HEIGHT);
            int width = static_cast<int>(w * original_shape.width / IMG_WIDTH);
            int height = static_cast<int>(h * original_shape.height / IMG_HEIGHT);

            boxes.push_back(cv::Rect(left, top, width, height));
            confidences.push_back(confidence);
            
            // Find the class with the highest score
            int class_id = 0;
            float max_score = 0.0;
            for(int j = 5; j < cols; ++j)
            {
                if(detections[i * cols + j] > max_score)
                {
                    max_score = detections[i * cols + j];
                    class_id = j - 5;
                }
            }
            class_ids.push_back(class_id);
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<TrackingBox> det_boxes;
    for (int idx : indices)
    {
        TrackingBox tb;
        tb.box = boxes[idx];
        tb.class_id = class_ids[idx];
        det_boxes.push_back(tb);
    }

    return tracker.update(det_boxes);
}

int main(int argc, char* argv[])
{
    if (argc < 3)
    {
        std::cout << "Usage: " << argv[0] << " <model_path.xml> <video_path>" << std::endl;
        return -1;
    }

    std::string model_path = argv[1];
    std::string video_path = argv[2];

    // --- OpenVINO Initialization ---
    ov::Core core;
    auto model = core.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_shape({1, (unsigned long)IMG_HEIGHT, (unsigned long)IMG_WIDTH, 3});
    ppp.input().preprocess().resize(ov::preprocess::ResizeAlgorithm::LINEAR);
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    ov::CompiledModel compiled_model = core.compile_model(model, "CPU");
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // --- Video and SORT Initialization ---
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    Sort tracker(10, 5, 0.3);
    cv::Mat frame;
    int frame_count = 0;

    while (cap.read(frame))
    {
        cv::Mat blob = preprocess(frame);
        ov::Tensor input_tensor(compiled_model.input().get_element_type(), compiled_model.input().get_shape(), blob.data);
        infer_request.set_input_tensor(input_tensor);
        infer_request.infer();

        ov::Tensor output_tensor = infer_request.get_output_tensor();
        std::vector<TrackingBox> tracked_objects = postprocess(output_tensor, frame.size(), tracker);

        std::cout << "Frame: " << frame_count << std::endl;
        for (const auto& obj : tracked_objects)
        {
            std::cout << "  ID: " << obj.id 
                      << ", Class: " << obj.class_id 
                      << ", Box: [" << obj.box.x << ", " << obj.box.y 
                      << ", " << obj.box.width << ", " << obj.box.height << "]" 
                      << std::endl;
        }

        frame_count++;
    }

    cap.release();

    return 0;
}
