import lps_sp.acoustical.analysis as lps_analysis
import lps_ml.audio_processors as ml_procs
import lps_ml.core.processor as ml_proc
import lps_ml.core.cv as ml_cv
import lps_utils.quantities as lps_qty

def get_file_processor() -> ml_procs.SampleProcessor:

    params = lps_analysis.Parameters(
                n_spectral_pts=4096,
                overlap=0,
                n_mels=512,
                decimation_rate=1,
                log_scale=True
            )

    analysis = lps_analysis.SpectralAnalysis.MELGRAM

    audio_pipelines : list[ml_proc.AudioPipeline] = [
        ml_procs.ToFloatConverter(),
        ml_procs.Resampler(lps_qty.Frequency.khz(16)),
        ml_procs.SpectralProcessor(analysis=analysis, params=params)
    ]

    n_samples=2**17
    overlap=2**16

    n_samples=int(n_samples / params.n_spectral_pts / 2)
    overlap=int(overlap / params.n_spectral_pts / 2)

    file_processor = ml_procs.SampleProcessor(
        n_samples=n_samples,
        overlap=overlap,
        audio_pipelines=audio_pipelines
    )

    return file_processor

def get_cv() -> ml_cv.CrossValidator:
    return ml_cv.FiveByTwo()
