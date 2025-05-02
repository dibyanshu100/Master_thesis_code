## Loss Functions in Diffusion Models: A Comparative Study

Diffusion models have established themselves as highly effective generative frameworks, inspiring significant research into their underlying mechanisms. An important aspect of these models lies in the choice of loss functions, which directly influences their training and performance. In this study we provide a comprehensive exploration of these loss functions, systematically analyzing their theoretical relationships and unifying them under the framework of the variational lower bound objective. We complement this analysis with empirical studies that examine the conditions under which different objectives yield varying performance and provide insights into the factors driving these discrepancies. Additionally, we assess the impact of loss function selection on the model’s ability to achieve specific objectives, such as producing high-quality samples or precisely estimating data likelihoods.By presenting a unified perspective, this study advances the understanding of loss functions in diffusion models, contributing to more efficient and goal-oriented model designs in future research.

The table below provide an overview of all the loss formulations across different scenarios. While the NELBO and the rescaled loss are equivalent and comparable, the weighted
losses are not equivalent and are expected to exhibit different empirical performance.


