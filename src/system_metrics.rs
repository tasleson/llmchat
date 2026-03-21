use serde::{Deserialize, Serialize};

#[cfg(all(target_os = "macos", feature = "macmon"))]
use std::sync::{Arc, Mutex};
#[cfg(all(target_os = "macos", feature = "macmon"))]
use std::thread;

#[cfg(all(target_os = "macos", feature = "macmon"))]
use macmon::{Metrics, Sampler, SocInfo};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreMetrics {
    pub freq_mhz_min: u32,
    pub freq_mhz_mean: f32,
    pub freq_mhz_median: f32,
    pub freq_mhz_max: u32,
    pub usage_percent_min: f32,
    pub usage_percent_mean: f32,
    pub usage_percent_median: f32,
    pub usage_percent_max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    pub ram_usage_gb_min: f32,
    pub ram_usage_gb_mean: f32,
    pub ram_usage_gb_median: f32,
    pub ram_usage_gb_max: f32,
    pub swap_usage_gb_min: f32,
    pub swap_usage_gb_mean: f32,
    pub swap_usage_gb_median: f32,
    pub swap_usage_gb_max: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PowerMetrics {
    pub cpu_watts_min: f32,
    pub cpu_watts_mean: f32,
    pub cpu_watts_median: f32,
    pub cpu_watts_max: f32,
    pub cpu_watts_total: f32,
    pub gpu_watts_min: f32,
    pub gpu_watts_mean: f32,
    pub gpu_watts_median: f32,
    pub gpu_watts_max: f32,
    pub gpu_watts_total: f32,
    pub ane_watts_min: f32,
    pub ane_watts_mean: f32,
    pub ane_watts_median: f32,
    pub ane_watts_max: f32,
    pub ane_watts_total: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetricsStats {
    pub efficiency_cores: CoreMetrics,
    pub performance_cores: CoreMetrics,
    pub gpu: CoreMetrics,
    pub memory: MemoryMetrics,
    pub power: PowerMetrics,
    pub sample_count: usize,
    pub duration_seconds: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_info: Option<HardwareInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub mac_model: String,
    pub chip_name: String,
    pub memory_gb: u8,
    pub ecpu_cores: u8,
    pub pcpu_cores: u8,
    pub ecpu_freqs: Vec<u32>,
    pub pcpu_freqs: Vec<u32>,
    pub gpu_cores: u8,
    pub gpu_freqs: Vec<u32>,
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
impl From<&SocInfo> for HardwareInfo {
    fn from(soc: &SocInfo) -> Self {
        HardwareInfo {
            mac_model: soc.mac_model.clone(),
            chip_name: soc.chip_name.clone(),
            memory_gb: soc.memory_gb,
            ecpu_cores: soc.ecpu_cores,
            pcpu_cores: soc.pcpu_cores,
            ecpu_freqs: soc.ecpu_freqs.clone(),
            pcpu_freqs: soc.pcpu_freqs.clone(),
            gpu_cores: soc.gpu_cores,
            gpu_freqs: soc.gpu_freqs.clone(),
        }
    }
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
pub struct MetricsMonitor {
    samples: Arc<Mutex<Vec<Metrics>>>,
    running: Arc<Mutex<bool>>,
    thread_handle: Option<thread::JoinHandle<()>>,
    soc_info: Option<SocInfo>,
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
impl MetricsMonitor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self {
            samples: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(Mutex::new(false)),
            thread_handle: None,
            soc_info: None,
        })
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        *self.running.lock().unwrap() = true;

        let samples = Arc::clone(&self.samples);
        let running = Arc::clone(&self.running);

        let handle = thread::spawn(move || {
            let mut sampler = match Sampler::new() {
                Ok(s) => s,
                Err(_) => return,
            };

            while *running.lock().unwrap() {
                match sampler.get_metrics(1000) {
                    Ok(metrics) => {
                        samples.lock().unwrap().push(metrics);
                    }
                    Err(_) => break,
                }
            }
        });

        // Capture hardware info from a temporary sampler
        match Sampler::new() {
            Ok(sampler) => {
                let soc = sampler.get_soc_info();
                // Only store if we got valid data
                if soc.ecpu_cores > 0 || soc.pcpu_cores > 0 {
                    self.soc_info = Some(soc.clone());
                } else {
                    eprintln!("Warning: Could not retrieve valid CPU core information from macmon");
                }
            }
            Err(e) => {
                eprintln!("Warning: Failed to create sampler for hardware info: {}", e);
            }
        }

        self.thread_handle = Some(handle);
        Ok(())
    }

    pub fn stop(&mut self) -> Result<SystemMetricsStats, Box<dyn std::error::Error>> {
        *self.running.lock().unwrap() = false;

        if let Some(handle) = self.thread_handle.take() {
            handle.join().map_err(|_| "Failed to join metrics thread")?;
        }

        let samples = self.samples.lock().unwrap();
        let mut stats = calculate_stats(&samples);
        stats.hardware_info = self.soc_info.as_ref().map(|soc| soc.into());
        Ok(stats)
    }
}

#[cfg(not(all(target_os = "macos", feature = "macmon")))]
pub struct MetricsMonitor;

#[cfg(not(all(target_os = "macos", feature = "macmon")))]
impl MetricsMonitor {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        Ok(Self)
    }

    pub fn start(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    pub fn stop(&mut self) -> Result<SystemMetricsStats, Box<dyn std::error::Error>> {
        Err("System metrics monitoring is only available on macOS".into())
    }
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn calculate_stats(samples: &[Metrics]) -> SystemMetricsStats {
    if samples.is_empty() {
        return SystemMetricsStats {
            efficiency_cores: CoreMetrics {
                freq_mhz_min: 0,
                freq_mhz_mean: 0.0,
                freq_mhz_median: 0.0,
                freq_mhz_max: 0,
                usage_percent_min: 0.0,
                usage_percent_mean: 0.0,
                usage_percent_median: 0.0,
                usage_percent_max: 0.0,
            },
            performance_cores: CoreMetrics {
                freq_mhz_min: 0,
                freq_mhz_mean: 0.0,
                freq_mhz_median: 0.0,
                freq_mhz_max: 0,
                usage_percent_min: 0.0,
                usage_percent_mean: 0.0,
                usage_percent_median: 0.0,
                usage_percent_max: 0.0,
            },
            gpu: CoreMetrics {
                freq_mhz_min: 0,
                freq_mhz_mean: 0.0,
                freq_mhz_median: 0.0,
                freq_mhz_max: 0,
                usage_percent_min: 0.0,
                usage_percent_mean: 0.0,
                usage_percent_median: 0.0,
                usage_percent_max: 0.0,
            },
            memory: MemoryMetrics {
                ram_usage_gb_min: 0.0,
                ram_usage_gb_mean: 0.0,
                ram_usage_gb_median: 0.0,
                ram_usage_gb_max: 0.0,
                swap_usage_gb_min: 0.0,
                swap_usage_gb_mean: 0.0,
                swap_usage_gb_median: 0.0,
                swap_usage_gb_max: 0.0,
            },
            power: PowerMetrics {
                cpu_watts_min: 0.0,
                cpu_watts_mean: 0.0,
                cpu_watts_median: 0.0,
                cpu_watts_max: 0.0,
                cpu_watts_total: 0.0,
                gpu_watts_min: 0.0,
                gpu_watts_mean: 0.0,
                gpu_watts_median: 0.0,
                gpu_watts_max: 0.0,
                gpu_watts_total: 0.0,
                ane_watts_min: 0.0,
                ane_watts_mean: 0.0,
                ane_watts_median: 0.0,
                ane_watts_max: 0.0,
                ane_watts_total: 0.0,
            },
            sample_count: 0,
            duration_seconds: 0.0,
            hardware_info: None,
        };
    }

    let calc_core_stats = |values: &[(u32, f32)]| -> CoreMetrics {
        let mut freqs: Vec<u32> = values.iter().map(|v| v.0).collect();
        let mut usages: Vec<f32> = values.iter().map(|v| v.1 * 100.0).collect();

        freqs.sort_unstable();
        usages.sort_by(|a, b| a.partial_cmp(b).unwrap());

        CoreMetrics {
            freq_mhz_min: *freqs.first().unwrap_or(&0),
            freq_mhz_mean: freqs.iter().sum::<u32>() as f32 / freqs.len() as f32,
            freq_mhz_median: median_u32(&freqs),
            freq_mhz_max: *freqs.last().unwrap_or(&0),
            usage_percent_min: *usages.first().unwrap_or(&0.0),
            usage_percent_mean: usages.iter().sum::<f32>() / usages.len() as f32,
            usage_percent_median: median_f32(&usages),
            usage_percent_max: *usages.last().unwrap_or(&0.0),
        }
    };

    let ecpu_values: Vec<_> = samples.iter().map(|s| s.ecpu_usage).collect();
    let pcpu_values: Vec<_> = samples.iter().map(|s| s.pcpu_usage).collect();
    let gpu_values: Vec<_> = samples.iter().map(|s| s.gpu_usage).collect();

    let mut ram_usage_gb: Vec<f32> = samples
        .iter()
        .map(|s| s.memory.ram_usage as f32 / 1024.0 / 1024.0 / 1024.0)
        .collect();
    let mut swap_usage_gb: Vec<f32> = samples
        .iter()
        .map(|s| s.memory.swap_usage as f32 / 1024.0 / 1024.0 / 1024.0)
        .collect();
    let mut cpu_watts: Vec<f32> = samples.iter().map(|s| s.cpu_power).collect();
    let mut gpu_watts: Vec<f32> = samples.iter().map(|s| s.gpu_power).collect();
    let mut ane_watts: Vec<f32> = samples.iter().map(|s| s.ane_power).collect();

    ram_usage_gb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    swap_usage_gb.sort_by(|a, b| a.partial_cmp(b).unwrap());
    cpu_watts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    gpu_watts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    ane_watts.sort_by(|a, b| a.partial_cmp(b).unwrap());

    SystemMetricsStats {
        efficiency_cores: calc_core_stats(&ecpu_values),
        performance_cores: calc_core_stats(&pcpu_values),
        gpu: calc_core_stats(&gpu_values),
        memory: MemoryMetrics {
            ram_usage_gb_min: min_f32(&ram_usage_gb),
            ram_usage_gb_mean: mean_f32(&ram_usage_gb),
            ram_usage_gb_median: median_f32(&ram_usage_gb),
            ram_usage_gb_max: max_f32(&ram_usage_gb),
            swap_usage_gb_min: min_f32(&swap_usage_gb),
            swap_usage_gb_mean: mean_f32(&swap_usage_gb),
            swap_usage_gb_median: median_f32(&swap_usage_gb),
            swap_usage_gb_max: max_f32(&swap_usage_gb),
        },
        power: PowerMetrics {
            cpu_watts_min: min_f32(&cpu_watts),
            cpu_watts_mean: mean_f32(&cpu_watts),
            cpu_watts_median: median_f32(&cpu_watts),
            cpu_watts_max: max_f32(&cpu_watts),
            cpu_watts_total: cpu_watts.iter().sum(),
            gpu_watts_min: min_f32(&gpu_watts),
            gpu_watts_mean: mean_f32(&gpu_watts),
            gpu_watts_median: median_f32(&gpu_watts),
            gpu_watts_max: max_f32(&gpu_watts),
            gpu_watts_total: gpu_watts.iter().sum(),
            ane_watts_min: min_f32(&ane_watts),
            ane_watts_mean: mean_f32(&ane_watts),
            ane_watts_median: median_f32(&ane_watts),
            ane_watts_max: max_f32(&ane_watts),
            ane_watts_total: ane_watts.iter().sum(),
        },
        sample_count: samples.len(),
        duration_seconds: samples.len() as f64, // approximately 1 second per sample
        hardware_info: None,
    }
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn median_u32(values: &[u32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) as f32 / 2.0
    } else {
        values[mid] as f32
    }
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn median_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn mean_f32(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn min_f32(values: &[f32]) -> f32 {
    values.iter().cloned().fold(f32::INFINITY, f32::min)
}

#[cfg(all(target_os = "macos", feature = "macmon"))]
fn max_f32(values: &[f32]) -> f32 {
    values.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}
