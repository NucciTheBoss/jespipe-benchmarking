#!/usr/bin/env python3

import joblib
import matplotlib.pyplot as plt
from jespipe.plugin.clean.plotter import Plot
from jespipe.plugin.start import start


class HyperParamCompare(Plot):
    def __init__(self, parameters: dict) -> None:
        self.model_list = parameters["model_list"]
        self.plot_name = parameters["plot_name"]
        self.save_path = parameters["save_path"]

    def plot(self) -> None:
        rmse_data_list = list()

        for model in self.model_list:
            rmse_data = joblib.load(model + "/stat/rmse.pkl")
            model_name = model.split("/"); model_name = model_name[-1]
            rmse_data_list.append((model_name, rmse_data.get("0.0")))

        data_dict = self._dict_builder(rmse_data_list)
        names = list(data_dict.keys())
        values = list(data_dict.values())

        plt.figure()
        plt.xlabel("Models"); plt.ylabel("Root Mean Squared Error (RMSE)")
        plt.scatter(names, values, c="red")
        plt.xticks(rotation=45)
        plt.title(f"{self.plot_name}")
        plt.savefig(self.save_path + f"/{self.plot_name}-rmse-values.png")
        plt.close()

    def _dict_builder(self, tuple_list: list) -> dict:
        d = dict()
        for data in tuple_list:
            d.update({data[0]: data[1]["rmse"]})

        return d


if __name__ == "__main__":
    stage, parameters = start()

    if stage == "clean":
        plotter = HyperParamCompare(parameters)
        plotter.plot()
