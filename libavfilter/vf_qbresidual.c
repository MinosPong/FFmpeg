/*
 * Copyright (c) 2012-2014 Clément Bœsch <u pkh me>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * QBResidual filter
 *
 * @see https://en.wikipedia.org/wiki/Canny_edge_detector
 */

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/lfg.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"

#include "dnn_interface.h"
#include <tensorflow/c/c_api.h>

struct plane_info {
    int8_t   *residual;
    int      width, height;
};

typedef struct QBResidualContext {
    const AVClass *class;
    
    char              *model_filename;
    DNNBackendType     backend_type;
    DNNModule         *dnn_module;
    DNNModel          *model;
    DNNInputData       input;
    DNNData            output;

    struct plane_info planes[3];
    int nb_planes;
    int mode;
    AVLFG                                   prng;
} QBResidualContext;

#define OFFSET(x) offsetof(QBResidualContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM
static const AVOption qbresidual_options[] = {
    { "dnn_backend", "DNN backend used for model execution", OFFSET(backend_type), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1, FLAGS, "backend" },
    { "native", "native backend flag", 0, AV_OPT_TYPE_CONST, { .i64 = 0 }, 0, 0, FLAGS, "backend" },
#if (CONFIG_LIBTENSORFLOW == 1)
    { "tensorflow", "tensorflow backend flag", 0, AV_OPT_TYPE_CONST, { .i64 = 1 }, 0, 0, FLAGS, "backend" },
#endif
    { "model", "path to model file specifying network architecture and its parameters", OFFSET(model_filename), AV_OPT_TYPE_STRING, {.str=NULL}, 0, 0, FLAGS },
    { NULL }
};


AVFILTER_DEFINE_CLASS(qbresidual);

static av_cold int init(AVFilterContext *ctx)
{
    QBResidualContext *qbresidual = ctx->priv;

    av_lfg_init(&qbresidual->prng, 0);

    qbresidual->input.dt = DNN_FLOAT;
    qbresidual->dnn_module = ff_get_dnn_module(qbresidual->backend_type);
    if (!qbresidual->dnn_module) {
        av_log(ctx, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        return AVERROR(ENOMEM);
    }
    if (!qbresidual->model_filename) {
        av_log(ctx, AV_LOG_ERROR, "model file for network is not specified\n");
        return AVERROR(EINVAL);
    }
    if (!qbresidual->dnn_module->load_model) {
        av_log(ctx, AV_LOG_ERROR, "load_model for network is not specified\n");
        return AVERROR(EINVAL);
    }

    qbresidual->model = (qbresidual->dnn_module->load_model)(qbresidual->model_filename);
    if (!qbresidual->model) {
        av_log(ctx, AV_LOG_ERROR, "could not load DNN model\n");
        return AVERROR(EINVAL);
    }

    return 0;
}

static int query_formats(AVFilterContext *context)
{
    const enum AVPixelFormat pixel_formats[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_NONE};

    AVFilterFormats *formats_list;

    formats_list = ff_make_format_list(pixel_formats);
    if (!formats_list){
        av_log(context, AV_LOG_ERROR, "could not create formats list\n");
        return AVERROR(ENOMEM);
    }

    return ff_set_common_formats(context, formats_list);
}

static int config_props(AVFilterLink *inlink)
{
    int p;
    AVFilterContext *ctx = inlink->dst;
    QBResidualContext *qbresidual = ctx->priv;
    //configure tensorflow
    const char *model_output_name = "y";
    DNNReturnType result;

    qbresidual->input.width    = inlink->w;
    qbresidual->input.height   = inlink->h;
    qbresidual->input.channels = 3;

    result = (qbresidual->model->set_input_output)(qbresidual->model->model, &qbresidual->input, "x", &model_output_name, 1);
    if (result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "could not set input and output for the model\n");
        return AVERROR(EIO);
    }

    //configure residual
    qbresidual->nb_planes = av_pix_fmt_count_planes(inlink->format);
    if (qbresidual->nb_planes != 3) {
        av_log(ctx, AV_LOG_ERROR, "Incorrect number of plane. It should be 3 but got %d\n", qbresidual->nb_planes);
        return AVERROR(EIO);
    }

    for (p = 0; p < qbresidual->nb_planes; p++) {
        struct plane_info *plane = &qbresidual->planes[p];

        plane->width      = inlink->w;
        plane->height     = inlink->h;
        plane->residual   = av_malloc(plane->width * plane->height);
        if (!plane->residual)
            return AVERROR(ENOMEM);
    }
    return 0;
}


/* 
static int32_t RESRANDOM(AVLFG *rand_state){
    return (int32_t)av_lfg_get(rand_state) % 30;
}

static void random_residual(AVLFG *rand_state, int w, int h,
                                uint8_t *dst, int dst_linesize,
                          const uint8_t *src, int src_linesize)
{
    int i,j;
    for (j = 1; j < h - 1; j++) {
        dst += dst_linesize;
        for (i = 1; i < w - 1; i++) {
            dst[i] = RESRANDOM(rand_state)
        }
    }
}*/


static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    QBResidualContext *qbresidual = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    int p = 0;
    AVFrame *out, *frame;

    //create out frame
    if (av_frame_is_writable(in)) {
        out = in;
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out) {
            av_frame_free(&in);
            return AVERROR(ENOMEM);
        }
        av_frame_copy_props(out, in);
    }

    frame = in;
    av_log(ctx, AV_LOG_INFO,
            "n:%4"PRId64" pos:%9"PRId64" "
            "s:%dx%d ",
            inlink->frame_count_out,
            frame->pkt_pos,
            frame->width, frame->height);

    //clear up
    if (out != in)
        av_frame_free(&in);
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    int p;
    QBResidualContext *qbresidual = ctx->priv;

    for (p = 0; p < qbresidual->nb_planes; p++) {
        struct plane_info *plane = &qbresidual->planes[p];
        av_freep(&plane->residual);
    }

    if (qbresidual->dnn_module) {
        (qbresidual->dnn_module->free_model)(&qbresidual->model);
        av_freep(&qbresidual->dnn_module);
    }
}

static const AVFilterPad qbresidual_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props,
        .filter_frame = filter_frame,
    },
    { NULL }
};

static const AVFilterPad qbresidual_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

AVFilter ff_vf_qbresidual = {
    .name          = "qbresidual",
    .description   = NULL_IF_CONFIG_SMALL("Apply residual filter."),
    .priv_size     = sizeof(QBResidualContext),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .inputs        = qbresidual_inputs,
    .outputs       = qbresidual_outputs,
    .priv_class    = &qbresidual_class,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
