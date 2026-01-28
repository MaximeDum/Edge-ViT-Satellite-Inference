#include <onnxruntime_cxx_api.h>
#include <iostream>

int main()
{
    try
    {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ViT_Inference");
        Ort::SessionOptions session_options;

        const char *model_path = "models/segformer_vit.onnx";

        std::cout << "Tentative de chargement du modèle : " << model_path << "..." << std::endl;
        Ort::Session session(env, model_path, session_options);

        std::cout << "Succès ! Le modèle ViT est chargé en C++." << std::endl;
    }
    catch (const Ort::Exception &e)
    {
        std::cerr << "Erreur lors du chargement : " << e.what() << std::endl;
        return 1;
    }
    return 0;
}