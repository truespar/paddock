use super::*;
use objc2_metal::MTLBuffer;

pub(super) fn mm(
    c: &Commands<'_>,
    m: &Matrix,
    b: &Buffer,
    x: &Buffer,
    out: &Buffer,
    rows: usize,
    mode: u32,
) {
    c.dispatch(
        "sam3_mm",
        &[&m.w, x, out, b],
        &[m.k as u32, m.n as u32, rows as u32, mode],
        [m.n.div_ceil(64), rows.div_ceil(32), 1],
        128,
    );
}
fn norm(
    c: &Commands<'_>,
    x: &Buffer,
    add: &Buffer,
    bias: &Buffer,
    n: &Norm,
    out: &Buffer,
    residual: bool,
) {
    c.dispatch(
        "sam3_norm",
        &[x, add, bias, &n.w, &n.b, out],
        &[1024, u32::from(residual)],
        [5184, 1, 1],
        256,
    );
}
impl Sam3Vision {
    /// Decode JPEG/PNG outside the model, then upload RGB at its original size.
    /// Filtering stays on the GPU; the result feeds the encoder without readback.
    pub fn encode_rgb(
        &mut self,
        rgb: &[u8],
        width: usize,
        height: usize,
        kind: Sam3InputKind,
        tracker: bool,
    ) -> Result<()> {
        self.encoded = false;
        self.tracker = false;
        self.input.land(
            &self.device,
            rgb,
            width,
            height,
            1008,
            kind,
            &self.ws.pixels,
        )?;
        self.encode_staged(kind == Sam3InputKind::VideoFrame, tracker)
    }
    /// Encode exactly one already-resized 1008-square RGB picture/frame.
    /// `video` selects Meta's FP16-stored video normalization, not image math.
    /// One fenced submission owns every scratch buffer; failures invalidate
    /// retained features before callers can accidentally reuse them.
    pub fn encode(&mut self, rgb: &[u8], video: bool, tracker: bool) -> Result<()> {
        self.encoded = false;
        self.tracker = false;
        if rgb.len() != 1008 * 1008 * 3 {
            return Err(error("SAM 3 encoder expects 1008x1008 RGB bytes"));
        }
        let w = &self.ws;
        // This model owns all submission and the previous call fenced it.
        unsafe {
            std::ptr::copy_nonoverlapping(
                rgb.as_ptr(),
                w.pixels.raw.contents().as_ptr().cast(),
                rgb.len(),
            );
        }
        self.encode_staged(video, tracker)
    }
    fn encode_staged(&mut self, video: bool, tracker: bool) -> Result<()> {
        let w = &self.ws;
        let c = self.device.begin()?;
        point(
            &c,
            "sam3_patch",
            &[&w.pixels, &w.patches],
            &[1008, 14, 24, 592, u32::from(video)],
            5184 * 592,
        );
        mm(&c, &self.patch, &self.pos, &w.patches, &w.x, 5184, 0);
        point(
            &c,
            "sam3_position",
            &[&w.x, &self.pos],
            &[5184 * 1024, 576 * 1024],
            5184 * 1024,
        );
        // ln_pre has a distinct F32 boundary before block 0's first LN.
        // Its result must be retained at F32, not replaced by half scratch.
        c.dispatch(
            "sam3_norm",
            &[
                &w.x,
                &w.projection,
                &self.norm.b,
                &self.norm.w,
                &self.norm.b,
                &w.x,
            ],
            &[1024, 2],
            [5184, 1, 1],
            256,
        );
        norm(
            &c,
            &w.x,
            &w.projection,
            &self.blocks[0].n1.b,
            &self.blocks[0].n1,
            &w.norm,
            false,
        );
        for (i, b) in self.blocks.iter().enumerate() {
            let global = usize::from(self.cfg.is_global(i));
            mm(&c, &b.qkv.w, &b.qkv.b, &w.norm, &w.qkv, 5184, 0);
            point(
                &c,
                "sam3_qkv",
                &[&w.qkv, &b.qkv.b, &self.rope[global], &w.q, &w.k, &w.v],
                &[5184, if global == 1 { 5184 } else { 576 }],
                5184 * 1024,
            );
            c.dispatch(
                "sam3_attention",
                &[&w.q, &w.k, &w.v, &w.attention, &self.tiles[global]],
                &[1],
                [16, 162, 1],
                64,
            );
            mm(&c, &b.out.w, &b.out.b, &w.attention, &w.projection, 5184, 1);
            norm(&c, &w.x, &w.projection, &b.out.b, &b.n2, &w.norm, true);
            mm(&c, &b.up.w, &b.up.b, &w.norm, &w.wide, 5184, 3);
            mm(&c, &b.down.w, &b.down.b, &w.wide, &w.projection, 5184, 1);
            let next = if i + 1 < self.blocks.len() {
                &self.blocks[i + 1].n1
            } else {
                &b.n2
            };
            norm(&c, &w.x, &w.projection, &b.down.b, next, &w.norm, true);
        }
        point(
            &c,
            "sam3_raster",
            &[&w.x, &w.raster],
            &[72, 24, 1024],
            5184 * 1024,
        );
        self.neck(&c, &self.det, &w.det);
        if tracker {
            self.tracker_neck_encoded(&c);
        }
        c.finish()?;
        self.encoded = true;
        self.tracker = tracker;
        Ok(())
    }
    fn neck(&self, c: &Commands<'_>, n: &Neck, out: &[Buffer; 3]) {
        let w = &self.ws;
        mm(c, &n.up4[0].w, &n.up4[0].b, &w.raster, &w.temp, 5184, 0);
        point(
            c,
            "sam3_convt",
            &[&w.temp, &n.up4[0].b, &w.half_a],
            &[72, 512, 1],
            144 * 144 * 512,
        );
        mm(
            c,
            &n.up4[1].w,
            &n.up4[1].b,
            &w.half_a,
            &w.temp,
            144 * 144,
            0,
        );
        point(
            c,
            "sam3_convt",
            &[&w.temp, &n.up4[1].b, &w.half_b],
            &[144, 256, 0],
            288 * 288 * 256,
        );
        self.neck_tail(c, n, 0, &w.half_b, &out[0]);
        mm(c, &n.up2.w, &n.up2.b, &w.raster, &w.temp, 5184, 0);
        point(
            c,
            "sam3_convt",
            &[&w.temp, &n.up2.b, &w.half_a],
            &[72, 512, 0],
            144 * 144 * 512,
        );
        self.neck_tail(c, n, 1, &w.half_a, &out[1]);
        self.neck_tail(c, n, 2, &w.raster, &out[2]);
    }
    fn neck_tail(&self, c: &Commands<'_>, n: &Neck, level: usize, input: &Buffer, out: &Buffer) {
        let side = 288 >> level;
        let a = &n.proj1[level];
        let b = &n.proj2[level];
        // temp is separate from the high-resolution input: no in-place GEMM.
        mm(c, &a.w, &a.b, input, &self.ws.temp, side * side, 2);
        c.dispatch(
            "sam3_conv3",
            &[&b.w.w, &self.ws.temp, out, &b.b],
            &[side as u32, 256],
            [4, (side * side).div_ceil(32), 1],
            128,
        );
    }
    fn tracker_neck_encoded(&self, c: &Commands<'_>) {
        let w = &self.ws;
        self.neck(c, &self.trk, &w.trk);
        for (i, a, out) in [(0, &self.s0, &w.s0), (1, &self.s1, &w.s1)] {
            let rows = (288 >> i) * (288 >> i);
            point(
                c,
                "grv_half",
                &[&w.trk[i], &w.half_b],
                &[(rows * 256) as u32],
                rows * 256,
            );
            mm(c, &a.w, &a.b, &w.half_b, out, rows, 4);
        }
    }
    pub fn ensure_tracker(&mut self) -> Result<()> {
        if !self.encoded {
            return Err(error("SAM 3 needs a successfully encoded picture"));
        }
        if !self.tracker {
            self.encoded = false;
            let c = self.device.begin()?;
            self.tracker_neck_encoded(&c);
            c.finish()?;
            self.encoded = true;
            self.tracker = true;
        }
        Ok(())
    }
    /// Fenced diagnostic readback. Serving heads will consume device planes.
    pub fn read_plane(&self, plane: Sam3VisionPlane) -> Result<Vec<f32>> {
        if !self.encoded {
            return Err(error("SAM 3 has no encoded picture"));
        }
        let (b, n, tracker) = match plane {
            Sam3VisionPlane::Trunk => (&self.ws.x, 5184 * 1024, false),
            Sam3VisionPlane::Detector(i) if i < 3 => {
                (&self.ws.det[i], (288 >> i) * (288 >> i) * 256, false)
            }
            Sam3VisionPlane::Tracker(i) if i < 3 => {
                (&self.ws.trk[i], (288 >> i) * (288 >> i) * 256, true)
            }
            Sam3VisionPlane::TrackerSkip0 => (&self.ws.s0, 288 * 288 * 32, true),
            Sam3VisionPlane::TrackerSkip1 => (&self.ws.s1, 144 * 144 * 64, true),
            _ => return Err(error("SAM 3 feature level must be 0, 1 or 2")),
        };
        if tracker && !self.tracker {
            return Err(error("SAM 3 tracker neck has not run"));
        }
        Ok(unsafe { b.read_f32(0, n) })
    }
}
