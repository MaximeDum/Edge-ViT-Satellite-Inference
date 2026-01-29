#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main(int argc, char *argv[])
{
    // 1. Vérification de l'argument
    if (argc < 2)
    {
        std::cout << "Usage: ./edge_vit_inference <path_to_image>" << std::endl;
        return -1;
    }

    std::string image_path = argv[1];

    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ViT_Inference");
        Ort::SessionOptions session_options;
        const char *model_path = "models/segformer_vit.onnx";
        Ort::Session session(env, model_path, session_options);

        // 2. Chargement de l'image passée en argument
        cv::Mat img = cv::imread(image_path);
        if (img.empty())
        {
            std::cerr << "Erreur : Impossible de lire l'image : " << image_path << std::endl;
            return -1;
        }

        cv::Mat resized_img, float_img;
        cv::resize(img, resized_img, cv::Size(512, 512));
        resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // Formatage NCHW
        std::vector<float> input_tensor_values(3 * 512 * 512);
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 512; h++)
            {
                for (int w = 0; w < 512; w++)
                {
                    input_tensor_values[c * 512 * 512 + h * 512 + w] = float_img.at<cv::Vec3f>(h, w)[2 - c];
                }
            }
        }

        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = {1, 3, 512, 512};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

        // 3. Inférence
        std::cout << "Inférence sur : " << image_path << "..." << std::endl;
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // 4. Post-processing et sauvegarde
        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        int out_h = (int)output_shape[2];
        int out_w = (int)output_shape[3];

        cv::Mat mask(out_h, out_w, CV_32FC1, output_data);
        cv::Mat final_mask;
        cv::normalize(mask, final_mask, 0, 255, cv::NORM_MINMAX, CV_8UC1);
        cv::resize(final_mask, final_mask, cv::Size(512, 512));

        std::string output_name = "mask_result.png";
        cv::imwrite(output_name, final_mask);
        std::cout << "Succès ! Masque sauvegardé sous : " << output_name << std::endl;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Erreur ONNX : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}