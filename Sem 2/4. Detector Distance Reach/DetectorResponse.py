"""Physical component response times, with explicit published/assumed provenance.

A ring-up time is an amplitude-envelope e-folding time, not a minimum source
lifetime. Mechanical, optical, receiver and propagation scales are kept separate.
The profiles do not reconstruct a calibrated complex GW transfer function from
an ASD, and must not be multiplied into an input-referred ASD a second time.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import json
import math
import warnings


class SteadyStateResponseWarning(UserWarning):
    """A monochromatic steady-state reach estimate needs additional validation."""


@dataclass(frozen=True)
class ResponseComponent:
    name: str
    kind: str
    parameters: dict
    origin: str
    scope: str
    source: str

    @property
    def amplitude_time_s(self):
        """Amplitude e-folding time; propagation windows have no such time."""
        p = self.parameters
        if self.kind == 'oscillator':
            omega = p.get('angular_frequency_rad_s')
            if omega is None:
                omega = 2 * math.pi * p['frequency_hz']
            if not (math.isfinite(omega) and omega > 0 and p['quality_factor'] > 0):
                raise ValueError(f'Invalid oscillator parameters for {self.name}')
            result = 2 * p['quality_factor'] / omega
        elif self.kind == 'lowpass':
            if not (math.isfinite(p['pole_hz']) and p['pole_hz'] > 0):
                raise ValueError(f'Invalid low-pass pole for {self.name}')
            result = 1 / (2 * math.pi * p['pole_hz'])
        elif self.kind == 'mechanical_damping':
            if not (math.isfinite(p['damping_rate_s_inv']) and p['damping_rate_s_inv'] > 0):
                raise ValueError(f'Invalid damping rate for {self.name}')
            result = 2 / p['damping_rate_s_inv']
        elif self.kind == 'propagation_window':
            return None
        else:
            raise ValueError(f'Unknown response component kind: {self.kind}')
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f'Invalid response parameters for {self.name}')
        return result

    @property
    def propagation_window_s(self):
        if self.kind != 'propagation_window':
            return None
        length, passes = self.parameters['length_m'], self.parameters['passes']
        if not (math.isfinite(length) and length > 0 and math.isfinite(passes) and passes > 0):
            raise ValueError(f'Invalid propagation window for {self.name}')
        return passes * length / 299792458.0

    def gw_resonance_hz(self, frequency_hz):
        """GW drive centre; it can differ from an electromagnetic carrier."""
        p = self.parameters
        if 'drive_comb_spacing_hz' in p:
            spacing = p['drive_comb_spacing_hz']
            return max(1, round(frequency_hz/spacing))*spacing
        return p.get('drive_frequency_hz', p.get('frequency_hz',
                     p.get('angular_frequency_rad_s', 0)/(2*math.pi)))

    @property
    def power_linewidth_hz(self):
        p = self.parameters
        return p.get('power_linewidth_hz',
                     p.get('frequency_hz',p.get('angular_frequency_rad_s',0)/(2*math.pi))/p['quality_factor'])

    def envelope_step(self, time_s):
        """Single-pole amplitude envelope after turn-on, at resonance if needed.

        This is a component-level check, not a finite-duration SNR correction.
        A detuned carrier, cascaded stages and collected ringdown need the full
        complex transfer function and an observation-window specification.
        """
        import numpy as np
        tau = self.amplitude_time_s
        if tau is None:
            raise ValueError('A propagation window is not an exponential ring-up')
        t = np.asarray(time_s, dtype=float)
        return -np.expm1(-np.maximum(t, 0) / tau)

    def envelope_transfer(self, envelope_frequency_hz):
        """First-order component envelope transfer, distinct from GW carrier H(f)."""
        import numpy as np
        tau = self.amplitude_time_s
        if tau is None:
            raise ValueError('Propagation windows need their finite-path kernel')
        return 1 / (1 + 2j*np.pi*np.asarray(envelope_frequency_hz)*tau)


@dataclass(frozen=True)
class DetectorResponse:
    detector_key: str
    name: str
    representative_component: str
    components: tuple
    assumptions: tuple
    limitations: tuple
    calibrated_full_transfer: bool = False

    @property
    def has_hypothetical_parameters(self):
        return any(c.origin == 'hypothetical' for c in self.components) or bool(self.assumptions)

    def component(self, name):
        try:
            return next(c for c in self.components if c.name == name)
        except StopIteration:
            raise ValueError(f'Unknown component {name!r} for {self.name}') from None

    def ring_up_time(self, component=None):
        """Return the selected component's amplitude time [s], not a hard cutoff."""
        selected = self.component(component or self.representative_component)
        result = selected.amplitude_time_s
        if result is None:
            raise ValueError('Light travel is a window/delay, not an exponential ring-up')
        return result

    def with_component_parameters(self, name, **parameters):
        original = self.component(name)
        updated = {**original.parameters, **parameters}
        if updated == original.parameters:
            return self
        override = replace(original, parameters=updated, origin='configured_override')
        return replace(self, components=tuple(override if c.name == name else c for c in self.components),
                       assumptions=(*self.assumptions,
                                    f'Configured parameter override for {name}: {parameters}'))

    def check_steady_state(self, frequency_hz, duration_s, strict=False):
        """Report limitations of h0*sqrt(T/Sn(f0)); do not suppress signal by fiat.

        Known slow poles matter near their resonances. An uncalibrated receiver
        or a hypothetical comb remains an explicit modelling assumption. The
        complete finite-duration waveform/noise integral is not implemented here.
        """
        if not (math.isfinite(frequency_hz) and frequency_hz > 0
                and math.isfinite(duration_s) and duration_s > 0):
            raise ValueError('Frequency and source duration must be finite and positive')
        issues = []
        selected = self.component(self.representative_component)
        if duration_s < 5 * selected.amplitude_time_s:
            # For an isolated mechanical resonance, do not impose its long time
            # on the far-off-resonance, broadband free-particle response.
            near = True
            if selected.kind == 'oscillator':
                f0 = selected.gw_resonance_hz(frequency_hz)
                width = selected.power_linewidth_hz
                near = abs(frequency_hz - f0) <= max(5*width, 1/duration_s)
            if near:
                issues.append('source may not reach the selected component\'s steady-state response')
        for c in self.components:
            if c.name == self.representative_component:
                continue
            if c.kind == 'lowpass' and duration_s < 5*c.amplitude_time_s:
                issues.append(f'source is short compared with {c.name}')
                continue
            if c.kind == 'propagation_window' and duration_s < c.propagation_window_s:
                issues.append(f'source is short compared with {c.name}')
                continue
            if c.kind != 'oscillator':
                continue
            f0 = c.gw_resonance_hz(frequency_hz)
            if duration_s < 5*c.amplitude_time_s and abs(frequency_hz-f0) <= max(5*c.power_linewidth_hz,1/duration_s):
                issues.append(f'short signal overlaps {c.name}')
        if self.has_hypothetical_parameters:
            issues.append('receiver/operating parameters include explicit benchmark assumptions')
        if issues:
            message = (f'{self.name}: ' + '; '.join(issues) + '. '
                       'The returned reach is a monochromatic steady-state approximation; '
                       'use a finite-duration waveform and a specified receiver response to validate it.')
            if strict:
                raise ValueError(message)
            warnings.warn(message, SteadyStateResponseWarning, stacklevel=3)
        return tuple(issues)


@lru_cache(maxsize=None)
def load_response_models(filepath=None):
    path = Path(filepath) if filepath else Path(__file__).parent/'data'/'detector_response_models.json'
    raw = json.loads(path.read_text())
    profiles = {}
    for key, row in raw['detectors'].items():
        components = tuple(ResponseComponent(**c) for c in row['components'])
        for component in components:
            if component.kind == 'propagation_window':
                component.propagation_window_s
            else:
                component.amplitude_time_s
        profiles[key] = DetectorResponse(
            detector_key=key, name=row['name'],
            representative_component=row['representative_component'],
            components=components, assumptions=tuple(row['assumptions']),
            limitations=tuple(row['limitations']),
            calibrated_full_transfer=row.get('calibrated_full_transfer', False))
        profiles[key].ring_up_time()  # Fail clearly on invalid/missing parameters.
    return profiles
