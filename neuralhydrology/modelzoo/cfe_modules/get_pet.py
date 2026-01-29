import torch

from neuralhydrology.modelzoo.cfe_modules.cfe_dataclasses import Flux


def daily_pet_jensen2016(T_avg: torch.Tensor, S_rad: torch.Tensor) -> torch.Tensor:
        """Calculate potential evapotranspiration (PET) using the Jensen et al. (2016) method.
        This method estimates PET from shortwave radiation and mean temperature.
    
        Parameters
        ----------
        T_avg : torch.Tensor
            Average temperature (°C)
        S_rad : torch.Tensor
            Shortwave radiation (W/m²)
        
        Outputs
        -------
        torch.Tensor
            Estimated potential evapotranspiration (PET) in mm/day.
        """
        shortRad = (
            S_rad * (24*3600) / 1000000
        ) # convert shortwave radiation [W/m^2] to [MJ/m^2 day]
        lambd = 2.501 - 0.002361 * T_avg  # using mean temp
        pet_mm_per_day = (0.025 * shortRad * (T_avg - (-3.0)) / lambd)
        pet_mm_per_day_mask = pet_mm_per_day < 0  # make mask for negative PET
        pet_mm_per_day = torch.where(pet_mm_per_day_mask, 0, pet_mm_per_day)  # clip negative PET to 0
        
        return pet_mm_per_day
