#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

int main()
{
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ViT_Inference");
        Ort::SessionOptions session_options;
        const char *model_path = "models/segformer_vit.onnx";
        Ort::Session session(env, model_path, session_options);

        // 1. Préparation de l'image
        cv::Mat img = cv::imread("satellite_test.jpg");
        if (img.empty())
        {
            std::cerr << "Image non trouvée !" << std::endl;
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

        // 2. Noms des entrées/sorties mis à jour
        const char *input_names[] = {"input"};
        const char *output_names[] = {"output"};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> input_dims = {1, 3, 512, 512};
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_dims.data(), input_dims.size());

        // 3. Inférence
        std::cout << "Lancement de l'inférence sur satellite_test.jpg..." << std::endl;
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // 4. Récupération et Post-processing
        float *output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();

        // Création du masque (SegFormer sort Batch x Classes x H x W)
        // Pour simplifier, on prend le premier canal (ou la classe dominante)
        int out_h = (int)output_shape[2];
        int out_w = (int)output_shape[3];

        cv::Mat mask(out_h, out_w, CV_32FC1, output_data);

        // Normalisation pour visualisation (0-255)
        cv::Mat final_mask;
        cv::normalize(mask, final_mask, 0, 255, cv::NORM_MINMAX, CV_8UC1);

        // Redimensionner à la taille d'entrée pour la comparaison
        cv::resize(final_mask, final_mask, cv::Size(512, 512));

        cv::imwrite("output_mask.png", final_mask);
        std::cout << "Succès ! Masque généré : output_mask.png (" << out_h << "x" << out_w << ")" << std::endl;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Erreur ONNX : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}