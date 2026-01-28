#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main()
{
    try
    {
        // 1. Setup ONNX Runtime
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ViT_Inference");
        Ort::SessionOptions session_options;
        const char *model_path = "models/segformer_vit.onnx";
        Ort::Session session(env, model_path, session_options);
        std::cout << "1. Modèle chargé." << std::endl;

        // 2. Charger l'image
        cv::Mat img = cv::imread("satellite_test.jpg");
        if (img.empty())
            return -1;

        // 3. Pré-traitement : Resize 512x512 et conversion Float
        cv::Mat resized_img, float_img;
        cv::resize(img, resized_img, cv::Size(512, 512));
        resized_img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

        // 4. Formatage NCHW (Planar RGB)
        // SegFormer attend : [Batch, Canaux, Hauteur, Largeur]
        size_t input_tensor_size = 3 * 512 * 512;
        std::vector<float> input_tensor_values(input_tensor_size);

        // On sépare les canaux d'OpenCV (BGR) pour les mettre dans le bon ordre (RGB) et en CHW
        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 512; h++)
            {
                for (int w = 0; w < 512; w++)
                {
                    // OpenCV est en BGR, on inverse (2-c) pour avoir du RGB
                    // Index = canal * (H*W) + ligne * W + colonne
                    input_tensor_values[c * 512 * 512 + h * 512 + w] =
                        float_img.at<cv::Vec3f>(h, w)[2 - c];
                }
            }
        }
        std::cout << "2. Formatage NCHW terminé." << std::endl;

        // 5. Création du tenseur ONNX (Input)
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_node_dims = {1, 3, 512, 512};

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_size,
            input_node_dims.data(), input_node_dims.size());

        std::cout << "3. Tenseur prêt pour l'inférence." << std::endl;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Erreur : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}