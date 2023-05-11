#pragma once
#include "AutoMorphingModel.h"
#include "Scene.h"

#include <memory>
#include <utility>

class BasicScene : public cg3d::Scene
{
public:
    explicit BasicScene(std::string name, cg3d::Display* display) : Scene(std::move(name), display) {};
    void Init(float fov, int width, int height, float near, float far);
    void Update(const cg3d::Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model) override;
    void MouseCallback(cg3d::Viewport* viewport, int x, int y, int button, int action, int mods, int buttonState[]) override;
    void ScrollCallback(cg3d::Viewport* viewport, int x, int y, int xoffset, int yoffset, bool dragging, int buttonState[]) override;
    void CursorPosCallback(cg3d::Viewport* viewport, int x, int y, bool dragging, int* buttonState)  override;
    void KeyCallback(cg3d::Viewport* viewport, int x, int y, int key, int scancode, int action, int mods) override;
    Eigen::Vector3f GetSpherePos();

    // New Functions
    Eigen::Vector3f GetDestinationPosition();
    Eigen::Vector3f GetLinkTipPosition(int link_id);
    Eigen::Vector3f GetLinkSourcePosition(int link_id);
    std::vector<Eigen::Matrix3f> GetEulerAnglesMatrices(Eigen::Matrix3f R);

    void IKCyclicCoordinateDecentMethod();
    void IKFabrikMethod();
    void IKSolverHelper(int link_id, Eigen::Vector3f D);

    // New Callback Functions
    void Space_Callback();
    void P_Callback();
    void T_Callback();
    void D_Callback();
    void N_Callback();
    void Right_Callback();
    void Left_Callback();
    void Up_Callback();
    void Down_Callback();
    void S_Callback();

    void Numbers_Callback(int num_of_link);

private:
    std::shared_ptr<Movable> root;
    std::shared_ptr<cg3d::Model> sphere1 ,cube;
    std::shared_ptr<cg3d::AutoMorphingModel> autoCube;
    std::vector<std::shared_ptr<cg3d::Model>> cyls, axis;
    int pickedIndex = 0;
    int tipIndex = 0;
    Eigen::VectorXi EMAP;
    Eigen::MatrixXi F,E,EF,EI;
    Eigen::VectorXi EQ;
    // If an edge were collapsed, we'd collapse it to these points:
    Eigen::MatrixXd V, C, N, T, points,edges,colors;

    // New Variables
    int IK_mode = 0;
    bool animate_CCD = false;
    bool animate_Fabrik = false;
    float delta = 0.05;
    float angle_divider = 50.f;

    int first_link_id = 0;
    int last_link_id = 2;
    int num_of_links = 3;
    float link_length = 1.6f;
};
