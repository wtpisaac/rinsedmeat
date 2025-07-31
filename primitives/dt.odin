package primitives

import "core:time"

_EWMASmoothingFactor :: distinct f64

EWMADt :: struct {
	smoothing_factor: _EWMASmoothingFactor,
	prev_tick:        Maybe(time.Tick),
	dt:               Maybe(f64),
}

EWMADtInitError :: enum {
	SMOOTHING_FACTOR_OUT_OF_RANGE,
}

// Initialize EWMADt with smoothing factor
@(require_results)
EWMADt_init :: proc(
	smoothing_factor: f64, // PRECONDITION: Smoothing factor expected between 0 and 1
) -> (
	dt: EWMADt,
	ok: Maybe(EWMADtInitError),
) {
	if (smoothing_factor < 0.0 || smoothing_factor > 1.0) {
		ok = .SMOOTHING_FACTOR_OUT_OF_RANGE
		return
	}

	dt = {
		smoothing_factor = _EWMASmoothingFactor(smoothing_factor),
	}
	return
}

EWMADt_record_tick :: proc(dt: ^EWMADt) {
	_EWMADt_record_tick(dt, time.tick_now())
}

_EWMADt_record_tick :: proc(dt: ^EWMADt, tick: time.Tick) {
	prev_tick, has_prev_tick := dt.prev_tick.?
	if (!has_prev_tick) {
		// We do not have any previous recorded tick, so this is the first.
		prev_tick := tick
		return
	}

	prev_dt, has_prev_dt := dt.dt.?
	if (!has_prev_dt) {
		dt.dt = f64(time.tick_diff(prev_tick, tick)) * 0.000001 /* ms per ns */
		return
	}

	// If we have reached this line, we have both a previous tick and the previously reported dt
	// We may compute our newest duration and smooth it by the set factor.
}

// Return duration in millis
EWMADt_retrieve_millis :: proc(dt: ^EWMADt, default_fps: f64 = 60.0) -> f64 {
	duration, has_duration := dt.dt.?
	if (!has_duration) {
		// Return millis per fps
		return default_fps / 1000
	}
	return duration
}
