using Gen
using JSON

# 特徴データを読み込む
function load_features(filepath)
    return JSON.parsefile(filepath)
end

# 事前分布のパラメータをロードする関数
function load_prior_means(filepath)
    if isfile(filepath)
        return JSON.parsefile(filepath)
    else
        return Dict("color_mean" => 100.0, "shape_mean" => 5.0, "texture_mean" => 40.0)
    end
end

# 階層的ベイズモデルの定義
@gen function trash_model(color::Vector{Float64}, shape::Float64, texture::Float64, prior_means::Dict)
    # ゴミの種類: 0=缶, 1=瓶, 2=ペットボトル, 3=ティッシュ, 4=紙コップ, 5=未知のゴミ
    trash_type = @trace(categorical([0.2, 0.2, 0.2, 0.1, 0.1, 0.2]), :trash_type)

    # 事前分布の平均値
    color_mean = @trace(normal(prior_means["color_mean"], 50.0), :color_mean)
    shape_mean = @trace(normal(prior_means["shape_mean"], 2.0), :shape_mean)
    texture_mean = @trace(normal(prior_means["texture_mean"], 15.0), :texture_mean)

    # 観測データに対する尤度
    @trace(normal(mean(color), 30), :color_obs)
    @trace(normal(shape, 2), :shape_obs)
    @trace(normal(texture, 10), :texture_obs)

    return trash_type
end

# 推論関数
function infer_trash_type(features)
    prior_means = load_prior_means("updated_means.json")
    observations = Dict(
        :color_obs => mean(features["color"]),
        :shape_obs => features["shape"],
        :texture_obs => features["texture"]
    )

    # 重要度サンプリングで推論
    (traces, _) = importance_sampling(trash_model, (features["color"], features["shape"], features["texture"], prior_means), observations, 1000)

    best_trace, _ = argmax(traces)
    inferred_type = get_retval(best_trace)

    # 更新された平均値を取得
    updated_color_mean = get_choices(best_trace)[:color_mean]
    updated_shape_mean = get_choices(best_trace)[:shape_mean]
    updated_texture_mean = get_choices(best_trace)[:texture_mean]

    # 平均値を保存
    open("updated_means.json", "w") do f
        JSON.print(f, Dict(
            "color_mean" => updated_color_mean,
            "shape_mean" => updated_shape_mean,
            "texture_mean" => updated_texture_mean
        ))
    end

    # 分類結果を表示
    trash_types = ["Can", "Glass Bottle", "Plastic Bottle", "Tissue", "Paper Cup", "Unknown"]
    println(trash_types[inferred_type + 1])
end

# メイン関数
function main()
    features = load_features("features.json")
    infer_trash_type(features)
end

main()