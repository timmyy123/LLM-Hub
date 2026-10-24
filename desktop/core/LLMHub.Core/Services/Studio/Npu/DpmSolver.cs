namespace LLMHub.Services.Studio.Npu;

/// <summary>
/// DPM++ 2M with the Karras noise schedule (k-diffusion's sample_dpmpp_2m) for Stable Diffusion 1.x's eps-prediction
/// UNet: 1000 training steps, scaled-linear betas 0.00085 to 0.012.
/// </summary>
public sealed class DpmSolver
{
    private readonly double[] _logSigmas = new double[1000];

    public DpmSolver()
    {
        double cumprod = 1;
        var start = Math.Sqrt(0.00085);
        var end = Math.Sqrt(0.012);
        for (var i = 0; i < 1000; i++)
        {
            var beta = Math.Pow(start + (end - start) * i / 999.0, 2);
            cumprod *= 1 - beta;
            _logSigmas[i] = Math.Log(Math.Sqrt((1 - cumprod) / cumprod));
        }
    }

    public double SigmaMin => Math.Exp(_logSigmas[0]);
    public double SigmaMax => Math.Exp(_logSigmas[^1]);

    /// <summary>Karras et al. (rho = 7) sigmas for <paramref name="steps"/> steps, followed by 0.</summary>
    public double[] Sigmas(int steps)
    {
        var sigmas = new double[steps + 1];
        var minInv = Math.Pow(SigmaMin, 1 / 7.0);
        var maxInv = Math.Pow(SigmaMax, 1 / 7.0);
        for (var i = 0; i < steps; i++)
        {
            var ramp = steps == 1 ? 0 : i / (double)(steps - 1);
            sigmas[i] = Math.Pow(maxInv + ramp * (minInv - maxInv), 7);
        }
        return sigmas;
    }

    /// <summary>The (fractional) training timestep for a sigma, by interpolating log-sigma.</summary>
    public double Timestep(double sigma)
    {
        var log = Math.Log(sigma);
        if (log <= _logSigmas[0])
            return 0;
        if (log >= _logSigmas[^1])
            return 999;
        var i = 0;
        while (_logSigmas[i + 1] < log)
            i++;
        return i + (log - _logSigmas[i]) / (_logSigmas[i + 1] - _logSigmas[i]);
    }

    /// <summary>
    /// Runs the sampler from <paramref name="sigmas"/>[<paramref name="start"/>] to 0. <paramref name="denoise"/> maps
    /// (x, sigma) to the model's estimate of the clean latent.
    /// </summary>
    public static void Sample(float[] x, double[] sigmas, int start, Func<float[], double, float[]> denoise, Action<int>? step = null)
    {
        float[]? previous = null;
        for (var i = start; i < sigmas.Length - 1; i++)
        {
            var denoised = denoise(x, sigmas[i]);
            var sigma = sigmas[i];
            var next = sigmas[i + 1];
            if (next == 0)
            {
                Array.Copy(denoised, x, x.Length);
            }
            else
            {
                var t = -Math.Log(sigma);
                var tNext = -Math.Log(next);
                var h = tNext - t;
                var ratio = (float)(next / sigma);
                var mix = (float)(1 - Math.Exp(-h));
                if (previous is null)
                {
                    for (var j = 0; j < x.Length; j++)
                        x[j] = ratio * x[j] + mix * denoised[j];
                }
                else
                {
                    var hLast = t + Math.Log(sigmas[i - 1]);
                    var r = hLast / h;
                    var a = (float)(1 + 1 / (2 * r));
                    var b = (float)(1 / (2 * r));
                    for (var j = 0; j < x.Length; j++)
                        x[j] = ratio * x[j] + mix * (a * denoised[j] - b * previous[j]);
                }
            }
            previous = denoised;
            step?.Invoke(i - start + 1);
        }
    }
}
