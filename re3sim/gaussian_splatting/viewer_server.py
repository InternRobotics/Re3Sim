import sanic
from sanic.response import html, file
from os.path import isfile, join

app = sanic.Sanic("Viewer")

@app.route('/<path:path>')
async def catch_all(request, path):
    file_path = join('/home/pjlab/main/real2sim/gaussian-splatting', path)
    # 如果请求的是静态文件，则返回该文件
    if isfile(file_path):
        return await file(file_path, headers={"Access-Control-Allow-Origin": "*"}, status=200)
    # 否则，返回 Vue 应用的 index.html，让 Vue Router 处理路由+
    return await file('./splat/index.html', headers={"Access-Control-Allow-Origin": "*"}, status=200)
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
    