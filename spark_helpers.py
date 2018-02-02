#!/usr/bin/env python

def spark_model_saver(sparkmodelCV, path):
    """
    Helper function to save the best model from the CrossValidator
    model

    Parameters
    sparkmodel (spark.ml.sparkmodelCV): spark CrossValidation model
    with .bestmodel attr

    path(str): path to save the model

    Returns: "Success"
    """
    sparkmodelCV.bestModel.save(path)
    return "Success"

def param_writer(cv_info_dict, outfile):
    """
    Save parameters and rmse to a text file for all models in
    a cv_info_dict returned by get_cv_info

    Parameters
    cv_info_dict (dict): dictionary with information about the CV model
    outfile (str): path to save text file
    """
    with open(outfile, "w") as of:
        for i in cv_info_dict["param_map"]:
            of.write(f"params = {i[0]} \nrmse = {i[1]}\n\n")

def get_CV_info(cv_model):
    """
    Returns a dictionary of information from inside the cv_test_model

    Parameters
    cv_model (spark.ml.tuning CrossValidator): fit crossvalidator model
    save_path (str): local path to save dict

    Returns
    cv_info_dict (dict): dictionary with information about the CV model
    """
    cv_info_dict = {}
    cv_info_dict["best_model"] = cv_model.bestModel
    cv_info_dict["avg_metrics"] = cv_model.avgMetrics
    cv_info_dict["model_list"] = cv_model.getEstimatorParamMaps()
    cv_info_dict["param_map"] = list(zip(cv_info_dict["model_list"],
                                     cv_info_dict["avg_metrics"]))
    return cv_info_dict
