#include "BasicScene.h"
#include <Eigen/src/Core/Matrix.h>
#include <edges.h>
#include <memory>
#include <per_face_normals.h>
#include <read_triangle_mesh.h>
#include <utility>
#include <vector>
#include "GLFW/glfw3.h"
#include "Mesh.h"
#include "PickVisitor.h"
#include "Renderer.h"
#include "ObjLoader.h"
#include "IglMeshLoader.h"

#include "igl/per_vertex_normals.h"
#include "igl/per_face_normals.h"
#include "igl/unproject_onto_mesh.h"
#include "igl/edge_flaps.h"
#include "igl/loop.h"
#include "igl/upsample.h"
#include "igl/AABB.h"
#include "igl/parallel_for.h"
#include "igl/shortest_edge_and_midpoint.h"
#include "igl/circulation.h"
#include "igl/edge_midpoints.h"
#include "igl/collapse_edge.h"
#include "igl/edge_collapse_is_valid.h"
#include "igl/write_triangle_mesh.h"

// #include "AutoMorphingModel.h"

using namespace cg3d;

void BasicScene::Init(float fov, int width, int height, float near, float far)
{
    camera = Camera::Create("camera", fov, float(width) / height, near, far);
    
    AddChild(root = Movable::Create("root")); // a common (invisible) parent object for all the shapes
    auto daylight{std::make_shared<Material>("daylight", "shaders/cubemapShader")}; 
    daylight->AddTexture(0, "textures/cubemaps/Daylight Box_", 3);
    auto background{Model::Create("background", Mesh::Cube(), daylight)};
    AddChild(background);
    background->Scale(120, Axis::XYZ);
    background->SetPickable(false);
    background->SetStatic();

 
    auto program = std::make_shared<Program>("shaders/phongShader");
    auto program1 = std::make_shared<Program>("shaders/pickingShader");
    
    auto material{std::make_shared<Material>("material", program)}; // empty material
    auto material1{std::make_shared<Material>("material", program1)}; // empty material
    //SetNamedObject(cube, Model::Create, Mesh::Cube(), material, shared_from_this());
 
    material->AddTexture(0, "textures/box0.bmp", 2);
    auto sphereMesh{IglLoader::MeshFromFiles("sphere_igl", "data/sphere.obj")};
    auto cylMesh{IglLoader::MeshFromFiles("cyl_igl","data/zcylinder.obj")};
    auto cubeMesh{IglLoader::MeshFromFiles("cube_igl","data/cube_old.obj")};
    sphere1 = Model::Create("sphere",sphereMesh, material);    
    cube = Model::Create("cube", cubeMesh, material);
    
    // Axis
    Eigen::MatrixXd vertices(6,3);
    vertices << -1,0,0,1,0,0,0,-1,0,0,1,0,0,0,-1,0,0,1;
    Eigen::MatrixXi faces(3,2);
    faces << 0,1,2,3,4,5;
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Ones(6, 3);
    Eigen::MatrixXd textureCoords = Eigen::MatrixXd::Ones(6, 2);
    std::shared_ptr<Mesh> coordsys = std::make_shared<Mesh>("coordsys", vertices, faces, vertexNormals, textureCoords);
    axis.push_back(Model::Create("axis", coordsys, material1));
    axis[0]->mode = 1;   
    axis[0]->Scale(4,Axis::XYZ);
    //axis[0]->lineWidth = 5;
    root->AddChild(axis[0]);
    float scaleFactor = 1; 
    cyls.push_back(Model::Create("cyl", cylMesh, material));
    cyls[0]->Scale(scaleFactor,Axis::Z);
    cyls[0]->SetCenter(Eigen::Vector3f(0, 0, -0.8f*scaleFactor));
    root->AddChild(cyls[0]);

    for(int i = 1; i < 3; i++)
    { 
        cyls.push_back(Model::Create("cyl", cylMesh, material));
        cyls[i]->Scale(scaleFactor,Axis::Z);   
        cyls[i]->Translate(1.6f*scaleFactor,Axis::Z);
        cyls[i]->SetCenter(Eigen::Vector3f(0, 0, -0.8f*scaleFactor));
        cyls[i-1]->AddChild(cyls[i]);

        // Axis
        axis.push_back(Model::Create("axis", coordsys, material1));
        axis[i]->mode = 1;
        axis[i]->Scale(4, Axis::XYZ);
        cyls[i-1]->AddChild(axis[i]);
        axis[i]->Translate(0.8f*scaleFactor, Axis::Z);
    }
    cyls[0]->Translate({0, 0, 0.8f*scaleFactor});
    root->RotateByDegree(90, Eigen::Vector3f(-1, 0, 0));

    auto morphFunc = [](Model* model, cg3d::Visitor* visitor) {
      return model->meshIndex; //(model->GetMeshList())[0]->data.size()-1;
    };
    autoCube = AutoMorphingModel::Create(*cube, morphFunc);

  
    sphere1->showWireframe = true;
    autoCube->Translate({-6,0,0});
    autoCube->Scale(1.5f);
    sphere1->Translate({5,0,0});

    autoCube->showWireframe = true;
    camera->Translate(22, Axis::Z);
    root->AddChild(sphere1);
    //root->AddChild(cyl);
    root->AddChild(autoCube);
    //points = Eigen::MatrixXd::Ones(1,3);
    //edges = Eigen::MatrixXd::Ones(1,3);
    //colors = Eigen::MatrixXd::Ones(1,3);
    
    //cyl->AddOverlay({points,edges,colors},true);
    cube->mode = 1; 
    auto mesh = cube->GetMeshList();

    //autoCube->AddOverlay(points,edges,colors);
    //mesh[0]->data.push_back({V,F,V,E});
    int num_collapsed;

    // Function to reset original mesh and data structures
    V = mesh[0]->data[0].vertices;
    F = mesh[0]->data[0].faces;
    //igl::read_triangle_mesh("data/cube.off",V,F);
    igl::edge_flaps(F,E,EMAP,EF,EI);
    std::cout<< "vertices: \n" << V <<std::endl;
    std::cout<< "faces: \n" << F <<std::endl;
    
    std::cout<< "edges: \n" << E.transpose() <<std::endl;
    std::cout<< "edges to faces: \n" << EF.transpose() <<std::endl;
    std::cout<< "faces to edges: \n "<< EMAP.transpose()<<std::endl;
    std::cout<< "edges indices: \n" << EI.transpose() <<std::endl;

    // Small update to fix the models appearance
    autoCube->Translate({ 0,0,0 });
    sphere1->Translate({ 0,0,0 });
}

void BasicScene::Update(const Program& program, const Eigen::Matrix4f& proj, const Eigen::Matrix4f& view, const Eigen::Matrix4f& model)
{
    Scene::Update(program, proj, view, model);
    program.SetUniform4f("lightColor", 0.8f, 0.3f, 0.0f, 0.5f);
    program.SetUniform4f("Kai", 1.0f, 0.3f, 0.6f, 1.0f);
    program.SetUniform4f("Kdi", 0.5f, 0.5f, 0.0f, 1.0f);
    program.SetUniform1f("specular_exponent", 5.0f);
    program.SetUniform4f("light_position", 0.0, 15.0f, 0.0, 1.0f);
    //cyl->Rotate(0.001f, Axis::Y);
    cube->Rotate(0.1f, Axis::XYZ);

    IKCyclicCoordinateDecentMethod();
    IKFabrikMethod();
}

void BasicScene::MouseCallback(Viewport* viewport, int x, int y, int button, int action, int mods, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event

    if (action == GLFW_PRESS) { // default mouse button press behavior
        PickVisitor visitor;
        visitor.Init();
        renderer->RenderViewportAtPos(x, y, &visitor); // pick using fixed colors hack
        auto modelAndDepth = visitor.PickAtPos(x, renderer->GetWindowHeight() - y);
        renderer->RenderViewportAtPos(x, y); // draw again to avoid flickering
        pickedModel = modelAndDepth.first ? std::dynamic_pointer_cast<Model>(modelAndDepth.first->shared_from_this()) : nullptr;
        pickedModelDepth = modelAndDepth.second;
        camera->GetRotation().transpose();
        xAtPress = x;
        yAtPress = y;

        //if (pickedModel)
        //    debug("found ", pickedModel->isPickable ? "pickable" : "non-pickable", " model at pos ", x, ", ", y, ": ",
        //          pickedModel->name, ", depth: ", pickedModelDepth);
        //else
        //    debug("found nothing at pos ", x, ", ", y);

        if (pickedModel && !pickedModel->isPickable)
            pickedModel = nullptr; // for non-pickable models we need only pickedModelDepth for mouse movement calculations later

        if (pickedModel)
            pickedToutAtPress = pickedModel->GetTout();
        else
            cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::ScrollCallback(Viewport* viewport, int x, int y, int xoffset, int yoffset, bool dragging, int buttonState[])
{
    // note: there's a (small) chance the button state here precedes the mouse press/release event
    auto system = camera->GetRotation().transpose();
    if (pickedModel) {
        //pickedModel->TranslateInSystem(system, { 0, 0, -float(yoffset) });
        //pickedToutAtPress = pickedModel->GetTout();


        // When one link of the arm is picked and being translated move all the arm
        // accordingly.The arm must not break!
        // Change ScrollCallback callback to translate the picked object away and to the
        // camera(perpendicular to camera plane).When no object is picked translate the
        // whole scene.
        bool arm_selected = false;
        for (int i = 0; i < num_of_links && !arm_selected; i++) {
            if (pickedModel == cyls[i]) {
                cyls[first_link_id]->TranslateInSystem(system * cyls[first_link_id]->GetRotation(), { 0, 0, -float(yoffset) });
                pickedToutAtPress = pickedModel->GetTout();
                arm_selected = true;
            }
        }
        // None of the arms were selected
        if (!arm_selected) {
            pickedModel->TranslateInSystem(system * pickedModel->GetRotation(), { 0, 0, -float(yoffset) });
            pickedToutAtPress = pickedModel->GetTout();
        }
    } else {
        camera->TranslateInSystem(system, {0, 0, -float(yoffset)});
        cameraToutAtPress = camera->GetTout();
    }
}

void BasicScene::CursorPosCallback(Viewport* viewport, int x, int y, bool dragging, int* buttonState)
{
    if (dragging) {
        auto system = camera->GetRotation().transpose() * GetRotation();
        auto moveCoeff = camera->CalcMoveCoeff(pickedModelDepth, viewport->width);
        auto angleCoeff = camera->CalcAngleCoeff(viewport->width);
        if (pickedModel) {
            //pickedModel->SetTout(pickedToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE) {
                //pickedModel->TranslateInSystem(system, { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });


                // When one link of the arm is picked and being translated move all the arm
                // accordingly.The arm must not break!
                // Right mouse button will translate the whole scene or the picked object.
                bool arm_selected = false;
                for (int i = 0; i < num_of_links && !arm_selected; i++) {
                    if (pickedModel == cyls[i]) {
                        cyls[first_link_id]->TranslateInSystem(system * cyls[first_link_id]->GetRotation(), { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });
                        arm_selected = true;
                    }
                }
                // None of the arms were selected
                if (!arm_selected) {
                    pickedModel->TranslateInSystem(system * pickedModel->GetRotation(), { -float(xAtPress - x) / moveCoeff, float(yAtPress - y) / moveCoeff, 0 });
                }
            }
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                //pickedModel->RotateInSystem(system, float(xAtPress - x) / angleCoeff, Axis::Y);
                //pickedModel->RotateInSystem(system, float(yAtPress - y) / angleCoeff, Axis::X);


                // Left mouse button will rotate objects or the scene in the same manner of the arrows
                bool arm_selected = false;
                for (int i = 0; i < num_of_links && !arm_selected; i++) {
                    if (pickedModel == cyls[i]) {
                        Eigen::Matrix3f R, R_without_root, R_new;
                        Eigen::Vector3f euler_angles;

                        if (i == 0) {
                            R = pickedModel->GetRotation();
                            R_without_root = root->GetRotation().transpose() * R;
                            euler_angles = R_without_root.eulerAngles(2, 0, 2);
                        }
                        else {
                            R = pickedModel->GetRotation();
                            R_without_root = cyls[i - 1]->GetRotation().transpose() * R;
                            euler_angles = R_without_root.eulerAngles(2, 0, 2);
                        }

                        // Left-Right mouse movements
                        float z_angle = -float(xAtPress - x) / angleCoeff;

                        // Up-Down mouse movements
                        float x_angle = float(yAtPress - y) / angleCoeff;

                        Eigen::AngleAxisf phi(euler_angles[0] + z_angle, Eigen::Vector3f::UnitZ());
                        Eigen::AngleAxisf theta(euler_angles[1] + x_angle, Eigen::Vector3f::UnitX());
                        Eigen::AngleAxisf psi(euler_angles[2], Eigen::Vector3f::UnitZ());

                        // Calculate new rotation
                        if (i == 0) {
                            R_new = root->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                            pickedModel->Rotate(R.transpose() * R_new);
                        }
                        else {
                            R_new = cyls[i - 1]->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                            pickedModel->Rotate(R.transpose() * R_new);
                        }

                        arm_selected = true;
                    }
                }
                // None of the arms were selected
                if (!arm_selected) {
                    pickedModel->RotateInSystem(system * pickedModel->GetRotation(), -float(xAtPress - x) / angleCoeff, Axis::Y);
                    pickedModel->RotateInSystem(system * pickedModel->GetRotation(), -float(yAtPress - y) / angleCoeff, Axis::X);
                } 
            }
        } else {
            //camera->SetTout(cameraToutAtPress);
            if (buttonState[GLFW_MOUSE_BUTTON_RIGHT] != GLFW_RELEASE)
                root->TranslateInSystem(system, {-float(xAtPress - x) / moveCoeff/10.0f, float( yAtPress - y) / moveCoeff/10.0f, 0});
            if (buttonState[GLFW_MOUSE_BUTTON_MIDDLE] != GLFW_RELEASE)
                root->RotateInSystem(system, float(x - xAtPress) / 180.0f, Axis::Z);
            if (buttonState[GLFW_MOUSE_BUTTON_LEFT] != GLFW_RELEASE) {
                root->RotateInSystem(system, float(x - xAtPress) / angleCoeff, Axis::Y);
                root->RotateInSystem(system, float(y - yAtPress) / angleCoeff, Axis::X);
            }
        }
        xAtPress =  x;
        yAtPress =  y;
    }
}

void BasicScene::KeyCallback(Viewport* viewport, int x, int y, int key, int scancode, int action, int mods)
{
    auto system = camera->GetRotation().transpose();

    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        switch (key) // NOLINT(hicpp-multiway-paths-covered)
        {
            case GLFW_KEY_ESCAPE:
                glfwSetWindowShouldClose(window, GLFW_TRUE);
                break;
            //case GLFW_KEY_UP:
            //    cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::X);
            //    break;
            //case GLFW_KEY_DOWN:
            //    cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::X);
            //    break;
            //case GLFW_KEY_LEFT:
            //    cyls[pickedIndex]->RotateInSystem(system, 0.1f, Axis::Y);
            //    break;
            //case GLFW_KEY_RIGHT:
            //    cyls[pickedIndex]->RotateInSystem(system, -0.1f, Axis::Y);
            //    break;
            //case GLFW_KEY_W:
            //    camera->TranslateInSystem(system, {0, 0.1f, 0});
            //    break;
            //case GLFW_KEY_S:
            //    camera->TranslateInSystem(system, {0, -0.1f, 0});
            //    break;
            //case GLFW_KEY_A:
            //    camera->TranslateInSystem(system, {-0.1f, 0, 0});
            //    break;
            //case GLFW_KEY_D:
            //    camera->TranslateInSystem(system, {0.1f, 0, 0});
            //    break;
            case GLFW_KEY_B:
                camera->TranslateInSystem(system, {0, 0, 0.1f});
                break;
            case GLFW_KEY_F:
                camera->TranslateInSystem(system, {0, 0, -0.1f});
                break;
            //case GLFW_KEY_1:
            //    if(pickedIndex > 0)
            //      pickedIndex--;
            //    break;
            //case GLFW_KEY_2:
            //    if(pickedIndex < cyls.size()-1)
            //        pickedIndex++;
            //    break;
            //case GLFW_KEY_3:
            //    if(tipIndex >= 0)
            //    {
            //      if(tipIndex == cyls.size())
            //        tipIndex--;
            //      sphere1->Translate(GetSpherePos());
            //      tipIndex--;
            //    }
            //    break;
            //case GLFW_KEY_4:
            //    if(tipIndex < cyls.size())
            //    {
            //        if(tipIndex < 0)
            //          tipIndex++;
            //        sphere1->Translate(GetSpherePos());
            //        tipIndex++;
            //    }
            //    break;

            // New Keys
            case GLFW_KEY_SPACE: // IK solver
                Space_Callback();
                break;
            case GLFW_KEY_P: // Prints rotation matrices
                P_Callback();
                break;
            case GLFW_KEY_T: // Prints arms tip positions
                T_Callback();
                break;
            case GLFW_KEY_D: // Prints destination position
                D_Callback();
                break;
            case GLFW_KEY_N: // Pick the next link, or the first one in case the last link is picked
                N_Callback();
                break;
            case GLFW_KEY_RIGHT: // Rotates picked link around the previous link Y axis
                Right_Callback();
                break;
            case GLFW_KEY_LEFT: // Rotates picked link around the previous link Y axis
                Left_Callback();
                break;
            case GLFW_KEY_UP: // Rotates picked link around the current X axis
                Up_Callback();
                break;
            case GLFW_KEY_DOWN: // Rotates picked link around the current X axis
                Down_Callback();
                break;
            case GLFW_KEY_S: // Switch IK modes
                S_Callback();
                break;
            case GLFW_KEY_R: // Reset arms positions
                Numbers_Callback(num_of_links);
                break;
            case GLFW_KEY_1: // Build 1 arm
                Numbers_Callback(1);
                break;
            case GLFW_KEY_2: // Build 2 arms
                Numbers_Callback(2);
                break;
            case GLFW_KEY_3: // Build 3 arms
                Numbers_Callback(3);
                break;
            case GLFW_KEY_4: // Build 4 arms
                Numbers_Callback(4);
                break;
            case GLFW_KEY_5: // Build 5 arms
                Numbers_Callback(5);
                break;
            case GLFW_KEY_6: // Build 6 arms
                Numbers_Callback(6);
                break;
            case GLFW_KEY_7: // Build 7 arms
                Numbers_Callback(7);
                break;
            case GLFW_KEY_8: // Build 8 arms
                Numbers_Callback(8);
                break;
            case GLFW_KEY_9: // Build 9 arms
                Numbers_Callback(9);
                break;
            case GLFW_KEY_RIGHT_BRACKET: // Add 1 more arm
                Numbers_Callback(num_of_links + 1);
                break;
            case GLFW_KEY_LEFT_BRACKET: // Remove 1 arm
                if (num_of_links != 1) {
                    Numbers_Callback(num_of_links - 1);
                }
                break;
        }
    }
}

Eigen::Vector3f BasicScene::GetSpherePos()
{
      Eigen::Vector3f l = Eigen::Vector3f(1.6f,0,0);
      Eigen::Vector3f res;
      res = cyls[tipIndex]->GetRotation()*l;   
      return res;  
}

// New Functions
// Get the destination position (sphere1 center)
Eigen::Vector3f BasicScene::GetDestinationPosition()
{
    Eigen::Matrix4f destination_transform = sphere1->GetAggregatedTransform();
    Eigen::Vector3f destination_position = Eigen::Vector3f(destination_transform.col(3).x(), destination_transform.col(3).y(), destination_transform.col(3).z());

    return destination_position;
}

// Get the tip position of cyls[link_id]
Eigen::Vector3f BasicScene::GetLinkTipPosition(int link_id)
{
    Eigen::Vector3f cyl_length = Eigen::Vector3f(0, 0, 0.8f);

    Eigen::Matrix4f arm_transform = cyls[link_id]->GetAggregatedTransform();
    Eigen::Vector3f arm_center = Eigen::Vector3f(arm_transform.col(3).x(), arm_transform.col(3).y(), arm_transform.col(3).z());
    Eigen::Vector3f arm_tip_position = arm_center + cyls[link_id]->GetRotation() * cyl_length;

    return arm_tip_position;
}

// Get the source position of cyls[link_id]
Eigen::Vector3f BasicScene::GetLinkSourcePosition(int link_id) {
    Eigen::Vector3f cyl_length = Eigen::Vector3f(0, 0, 0.8f);

    Eigen::Matrix4f arm_transform = cyls[link_id]->GetAggregatedTransform();
    Eigen::Vector3f arm_center = Eigen::Vector3f(arm_transform.col(3).x(), arm_transform.col(3).y(), arm_transform.col(3).z());
    Eigen::Vector3f arm_source_position = arm_center - cyls[link_id]->GetRotation() * cyl_length;

    return arm_source_position;
}

// Get the euler matrices (A_0, A_1, A_2) according to ZXZ Euler angles
std::vector<Eigen::Matrix3f> BasicScene::GetEulerAnglesMatrices(Eigen::Matrix3f R) {
    // Get phi, theta and psi, according to ZXZ Euler angles
    Eigen::Vector3f zxz = R.eulerAngles(2, 0, 2);

    // Building euler angles matrices
    Eigen::Matrix3f phi;
    phi.row(0) = Eigen::Vector3f(cos(zxz.x()), -sin(zxz.x()), 0);
    phi.row(1) = Eigen::Vector3f(sin(zxz.x()), cos(zxz.x()), 0);
    phi.row(2) = Eigen::Vector3f(0, 0, 1);

    Eigen::Matrix3f theta;
    theta.row(0) = Eigen::Vector3f(1, 0, 0);
    theta.row(1) = Eigen::Vector3f(0, cos(zxz.y()), -sin(zxz.y()));
    theta.row(2) = Eigen::Vector3f(0, sin(zxz.y()), cos(zxz.y()));

    Eigen::Matrix3f psi;
    psi.row(0) = Eigen::Vector3f(cos(zxz.z()), -sin(zxz.z()), 0);
    psi.row(1) = Eigen::Vector3f(sin(zxz.z()), cos(zxz.z()), 0);
    psi.row(2) = Eigen::Vector3f(0, 0, 1);

    std::vector<Eigen::Matrix3f> euler_angles_matrices;
    euler_angles_matrices.push_back(phi);
    euler_angles_matrices.push_back(theta);
    euler_angles_matrices.push_back(psi);

    return euler_angles_matrices;
}

// Inverse Kinematics Coordinate Decent Method
void BasicScene::IKCyclicCoordinateDecentMethod() {
    if (animate_CCD && animate) {
        Eigen::Vector3f D = GetDestinationPosition();
        Eigen::Vector3f first_link_position = GetLinkSourcePosition(first_link_id);

        if ((D - first_link_position).norm() > link_length * num_of_links) {
            std::cout << "cannot reach" << std::endl;
            animate_CCD = false;
            return;
        }

        int curr_link = last_link_id;

        while (curr_link != -1) {
            Eigen::Vector3f R = GetLinkSourcePosition(curr_link);
            Eigen::Vector3f E = GetLinkTipPosition(last_link_id);
            Eigen::Vector3f RD = D - R;
            Eigen::Vector3f RE = E - R;
            float distance = (D - E).norm();

            if (distance < delta) {
                std::cout << "distance: " << distance << std::endl;
                animate_CCD = false;
                return;
            }

            // The plane normal
            Eigen::Vector3f normal = RE.normalized().cross(RD.normalized()); 

            // Get dot product
            float dot = RD.normalized().dot(RE.normalized());

            // Check that it is between -1 to 1
            if (dot > 1) dot = 1;
            if (dot < -1) dot = -1;

            // Rotate link
            float angle = acosf(dot) / angle_divider;
            Eigen::Vector3f rotation_vector = cyls[curr_link]->GetRotation().transpose() * normal;
            Eigen::Matrix3f Ri = (Eigen::AngleAxisf(angle, rotation_vector.normalized())).toRotationMatrix();
            Eigen::Vector3f euler_angles = Ri.eulerAngles(2, 0, 2);
             
            cyls[curr_link]->Rotate(euler_angles[0], Axis::Z);
            cyls[curr_link]->Rotate(euler_angles[1], Axis::X);
            cyls[curr_link]->Rotate(euler_angles[2], Axis::Z);

            curr_link--;
        }
        animate = false;
    }
}

// Inverse Kinematics Fabrik Method
void BasicScene::IKFabrikMethod() {
    if (animate_Fabrik && animate) {
        // The joint positions
        std::vector<Eigen::Vector3f> p; 
        p.resize(num_of_links + 1);

        // The target position
        Eigen::Vector3f t = GetDestinationPosition();

        // The root position
        Eigen::Vector3f root = GetLinkSourcePosition(first_link_id);

        // Set disjoint positions (p_0 the is first disjoin)
        int curr = first_link_id;
        while (curr != num_of_links) {
            p[curr] = GetLinkSourcePosition(curr);
            curr = curr + 1;
        }
        p[last_link_id + 1] = GetLinkTipPosition(last_link_id);

        std::vector<double> ri_array;
        std::vector<double> lambda_i_array;

        ri_array.resize(num_of_links + 1);
        lambda_i_array.resize(num_of_links + 1);

        // 1.1. % The distance between root and target 
        float dist = (root - t).norm();

        // 1.3. % Check whether the target is within reach
        if (dist > link_length * num_of_links) {
            // 1.5. % The target is unreachable
            std::cout << "cannot reach" << std::endl;
            animate_Fabrik = false;
            return;
        }
        else {
            // 1.14. % The target is reachable; thus set as b the initial position of joint p_0
            Eigen::Vector3f b = p[first_link_id];

            // 1.16. % Check wether the distance between the end effector p_n and the target t is greater then a tolerance
            Eigen::Vector3f endEffector = p[last_link_id + 1];
            float diff_A = (endEffector - t).norm();
            float tol = delta;

            if (diff_A < tol) {
                std::cout << "distance: " << diff_A << std::endl;
                animate_Fabrik = false;
                return;
            }
            while (diff_A > tol) {
                // 1.19. % STAGE 1: FORWARD REACHING
                // 1.20. % Set the end effector p_n as target t
                p[last_link_id + 1] = t;
                int parent = last_link_id;
                int child = last_link_id + 1;

                while (parent != -1) {
                    // 1.23. % Find the distance r_i between the new joint position p_i+1 and the joint p_i
                    ri_array[parent] = (p[child] - p[parent]).norm();
                    lambda_i_array[parent] = link_length / ri_array[parent];
                    // 1.26. % Find the new joint positions p_i.
                    p[parent] = (1 - lambda_i_array[parent]) * p[child] + lambda_i_array[parent] * p[parent];
                    child = parent;
                    parent = parent - 1;
                }
                // 1.29. % STAGE 2: BACKWORD REACHING
                // 1.30. % Set the root p0 its initial position
                p[first_link_id] = b;
                parent = first_link_id;
                child = first_link_id + 1;

                while (child != num_of_links) {
                    //1.33. % Find the distance r_i between the new joint position p_i and the joint p_i+1
                    ri_array[parent] = (p[child] - p[parent]).norm();
                    lambda_i_array[parent] = link_length / ri_array[parent];
                    //1.36 % Find the new joint positions p_i.
                    p[child] = (1 - lambda_i_array[parent]) * p[parent] + lambda_i_array[parent] * p[child];
                    parent = child;
                    child = child + 1;
                }                
                diff_A = (p[last_link_id + 1] - t).norm();
            }

            // Using Fabrik output to rotate the links
            int curr_link = first_link_id;
            int target_id = first_link_id + 1;

            while (curr_link != num_of_links) {
                IKSolverHelper(curr_link, p[target_id]);
                curr_link = target_id;
                target_id = target_id + 1;
            }

            float distance = (t - GetLinkTipPosition(last_link_id)).norm();

            if (distance < delta) {
                animate_Fabrik = false;
                std::cout << "distance: " << distance << std::endl;
            }
        }
        animate = false;
    }
}

// Inverse Kinematics helper function to perform the links rotations
void BasicScene::IKSolverHelper(int link_id, Eigen::Vector3f D) {
    Eigen::Vector3f R = GetLinkSourcePosition(link_id);
    Eigen::Vector3f E = GetLinkTipPosition(link_id);
    Eigen::Vector3f RD = D - R;
    Eigen::Vector3f RE = E - R;

    // The plane normal
    Eigen::Vector3f normal = RE.normalized().cross(RD.normalized());

    // Get dot product
    float dot = RD.normalized().dot(RE.normalized()); 

    // Check that it is between -1 to 1
    if (dot > 1) dot = 1;
    if (dot < -1) dot = 1;

    // Rotate link
    float angle = acosf(dot) / angle_divider;
    Eigen::Vector3f rotation_vector = cyls[link_id]->GetRotation().transpose() * normal;
    Eigen::Matrix3f Ri = (Eigen::AngleAxisf(angle, rotation_vector.normalized())).toRotationMatrix();
    Eigen::Vector3f euler_angles = Ri.eulerAngles(2, 0, 2);

    cyls[link_id]->Rotate(euler_angles[0], Axis::Z);
    cyls[link_id]->Rotate(euler_angles[1], Axis::X);
    cyls[link_id]->Rotate(euler_angles[2], Axis::Z);
}

// New Callback Functions
void BasicScene::Space_Callback()
{
    if (IK_mode == 0) {
        if (!animate_CCD) {
            animate_CCD = true;
        }
        else {
            animate_CCD = false;
        }
    }
    else {
        if (!animate_Fabrik) {
            animate_Fabrik = true;
        }
        else {
            animate_Fabrik = false;
        }
    }
}

void BasicScene::P_Callback()
{
    for (int i = 0; i < num_of_links; i++) {
        if (pickedModel == cyls[i]) {
            Eigen::Matrix3f arm_rotation = pickedModel->GetRotation();

            std::cout << "Arm" << i << " Rotation: " << std::endl
                << arm_rotation
                << std::endl
                << std::endl;

            Eigen::Vector3f arm_euler_angles = arm_rotation.eulerAngles(2, 0, 2) * (180.f / 3.14f);
            std::vector<Eigen::Matrix3f> euler_angles_matrices = GetEulerAnglesMatrices(arm_rotation);

            std::cout << "Arm" << i << " Euler Angles: " << std::endl
                << "phi: " << arm_euler_angles.x() << " (Deg)" << std::endl
                << "theta: " << arm_euler_angles.y() << " (Deg)" << std::endl
                << "psi: " << arm_euler_angles.z() << " (Deg)" << std::endl
                << std::endl;

            std::cout << "phi matrix: " << std::endl
                << euler_angles_matrices[0]
                << std::endl
                << std::endl;

            std::cout << "theta matrix: " << std::endl
                << euler_angles_matrices[1]
                << std::endl
                << std::endl;

            std::cout << "psi matrix: " << std::endl
                << euler_angles_matrices[2]
                << std::endl;

            return;
        }
    }
    // None of the arms were selected
    Eigen::Matrix3f scene_rotation = root->GetRotation();

    std::cout << "Scene Rotation: " << std::endl
        << scene_rotation << std::endl;
}

void BasicScene::T_Callback()
{
    for (int i = 0; i < num_of_links; i++) {
        Eigen::Vector3f arm_tip_position = GetLinkTipPosition(i);

        std::cout << "Arm" << i << " Tip Position: " << std::endl
            << arm_tip_position
            << std::endl
            << std::endl;
    }
}

void BasicScene::D_Callback() 
{
    Eigen::Vector3f destination_position = GetDestinationPosition();

    std::cout << "Destination Position: " << std::endl
        << destination_position
        << std::endl;
}

void BasicScene::N_Callback()
{
    bool arm_selected = false;
    for (int i = 0; i < num_of_links && !arm_selected; i++) {
        if (pickedModel == cyls[i]) {
            // Last link
            if (i == num_of_links - 1) {
                i = -1;
            }
            pickedModel = cyls[i + 1];
            arm_selected = true;
        }
    }
    // None of the arms were selected
    if (!arm_selected) {
        pickedModel = cyls[0];
    }
}

void BasicScene::Right_Callback()
{
    auto system = camera->GetRotation().transpose();

    bool arm_selected = false;
    for (int i = 0; i < num_of_links && !arm_selected; i++) {
        if (pickedModel == cyls[i]) {
            Eigen::Matrix3f R, R_without_root, R_new;
            Eigen::Vector3f euler_angles;

            if (i == 0) {
                R = pickedModel->GetRotation();
                R_without_root = root->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }
            else {
                R = pickedModel->GetRotation();
                R_without_root = cyls[i - 1]->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }

            float angle = 0.1f;
            Eigen::AngleAxisf phi(euler_angles[0] + angle, Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf theta(euler_angles[1], Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf psi(euler_angles[2], Eigen::Vector3f::UnitZ());

            // Calculate new rotation
            if (i == 0) {
                R_new = root->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }
            else {
                R_new = cyls[i - 1]->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }

            arm_selected = true;
        }
    }
    // None of the arms were selected
    if (!arm_selected) {
        root->RotateInSystem(system, -0.1f, Axis::Y);
    }
}

void BasicScene::Left_Callback()
{
    auto system = camera->GetRotation().transpose();

    bool arm_selected = false;
    for (int i = 0; i < num_of_links && !arm_selected; i++) {
        if (pickedModel == cyls[i]) {
            Eigen::Matrix3f R, R_without_root, R_new;
            Eigen::Vector3f euler_angles;

            if (i == 0) {
                R = pickedModel->GetRotation();
                R_without_root = root->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }
            else {
                R = pickedModel->GetRotation();
                R_without_root = cyls[i - 1]->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }

            float angle = -0.1f;
            Eigen::AngleAxisf phi(euler_angles[0] + angle, Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf theta(euler_angles[1], Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf psi(euler_angles[2], Eigen::Vector3f::UnitZ());

            // Calculate new rotation
            if (i == 0) {
                R_new = root->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }
            else {
                R_new = cyls[i - 1]->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }

            arm_selected = true;
        }
    }
    // None of the arms were selected
    if (!arm_selected) {
        root->RotateInSystem(system, 0.1f, Axis::Y);
    }
}

void BasicScene::Up_Callback()
{
    auto system = camera->GetRotation().transpose();

    bool arm_selected = false;
    for (int i = 0; i < num_of_links && !arm_selected; i++) {
        if (pickedModel == cyls[i]) {
            Eigen::Matrix3f R, R_without_root, R_new;
            Eigen::Vector3f euler_angles;

            if (i == 0) {
                R = pickedModel->GetRotation();
                R_without_root = root->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }
            else {
                R = pickedModel->GetRotation();
                R_without_root = cyls[i - 1]->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }

            float angle = 0.1f;
            Eigen::AngleAxisf phi(euler_angles[0], Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf theta(euler_angles[1] + angle, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf psi(euler_angles[2], Eigen::Vector3f::UnitZ());

            // Calculate new rotation
            if (i == 0) {
                R_new = root->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }
            else {
                R_new = cyls[i - 1]->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }

            arm_selected = true;
        }
    }
    // None of the arms were selected
    if (!arm_selected) {
        root->RotateInSystem(system, -0.1f, Axis::X);
    }
}

void BasicScene::Down_Callback()
{
    auto system = camera->GetRotation().transpose();

    bool arm_selected = false;
    for (int i = 0; i < num_of_links && !arm_selected; i++) {
        if (pickedModel == cyls[i]) {
            Eigen::Matrix3f R, R_without_root, R_new;
            Eigen::Vector3f euler_angles;

            if (i == 0) {
                R = pickedModel->GetRotation();
                R_without_root = root->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }
            else {
                R = pickedModel->GetRotation();
                R_without_root = cyls[i - 1]->GetRotation().transpose() * R;
                euler_angles = R_without_root.eulerAngles(2, 0, 2);
            }

            float angle = -0.1f;
            Eigen::AngleAxisf phi(euler_angles[0], Eigen::Vector3f::UnitZ());
            Eigen::AngleAxisf theta(euler_angles[1] + angle, Eigen::Vector3f::UnitX());
            Eigen::AngleAxisf psi(euler_angles[2], Eigen::Vector3f::UnitZ());

            // Calculate new rotation
            if (i == 0) {
                R_new = root->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }
            else {
                R_new = cyls[i - 1]->GetRotation() * Eigen::Quaternionf(phi * theta * psi).toRotationMatrix();
                pickedModel->Rotate(R.transpose() * R_new);
            }

            arm_selected = true;
        }
    }
    // None of the arms were selected
    if (!arm_selected) {
        root->RotateInSystem(system, 0.1f, Axis::X);
    }
}

void BasicScene::S_Callback()
{
    animate_CCD = false;
    animate_Fabrik = false;

    if (IK_mode == 1) {
        std::cout << "IK mode: Cyclic Coordinate Decent Method" << std::endl;
        IK_mode = 0;
    }
    else {
        std::cout << "IK mode: Fabrik Method" << std::endl;
        IK_mode = 1;
    }
}

void BasicScene::Numbers_Callback(int num_of_link) {
    animate_CCD = false;
    animate_Fabrik = false;
    root->RemoveChild(axis[0]);
    root->RemoveChild(cyls[0]);
    axis.clear();
    cyls.clear();
    pickedModel = NULL;


    auto program = std::make_shared<Program>("shaders/phongShader");
    auto program1 = std::make_shared<Program>("shaders/pickingShader");

    auto material{std::make_shared<Material>("material", program)}; // empty material
    auto material1{std::make_shared<Material>("material", program1)}; // empty material

    material->AddTexture(0, "textures/box0.bmp", 2);
    auto cylMesh{IglLoader::MeshFromFiles("cyl_igl","data/zcylinder.obj")};

    Eigen::MatrixXd vertices(6, 3);
    vertices << -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, -1, 0, 0, 1;
    Eigen::MatrixXi faces(3, 2);
    faces << 0, 1, 2, 3, 4, 5;
    Eigen::MatrixXd vertexNormals = Eigen::MatrixXd::Ones(6, 3);
    Eigen::MatrixXd textureCoords = Eigen::MatrixXd::Ones(6, 2);
    std::shared_ptr<Mesh> coordsys = std::make_shared<Mesh>("coordsys", vertices, faces, vertexNormals, textureCoords);
    axis.push_back(Model::Create("axis", coordsys, material1));
    axis[0]->mode = 1;
    axis[0]->Scale(4, Axis::XYZ);
    root->AddChild(axis[0]);
    float scaleFactor = 1;
    cyls.push_back(Model::Create("cyl", cylMesh, material));
    cyls[0]->Scale(scaleFactor, Axis::Z);
    cyls[0]->SetCenter(Eigen::Vector3f(0, 0, -0.8f * scaleFactor));
    root->AddChild(cyls[0]);

    for (int i = 1; i < num_of_link; i++)
    {
        cyls.push_back(Model::Create("cyl", cylMesh, material));
        cyls[i]->Scale(scaleFactor, Axis::Z);
        cyls[i]->Translate(1.6f * scaleFactor, Axis::Z);
        cyls[i]->SetCenter(Eigen::Vector3f(0, 0, -0.8f*scaleFactor));
        cyls[i-1]->AddChild(cyls[i]);

        // Axis
        axis.push_back(Model::Create("axis", coordsys, material1));
        axis[i]->mode = 1;
        axis[i]->Scale(4, Axis::XYZ);
        cyls[i-1]->AddChild(axis[i]);
        axis[i]->Translate(0.8f*scaleFactor, Axis::Z);
    }
    cyls[0]->Translate({0, 0, 0.8f*scaleFactor});
    root->Translate({0 ,0, 0});

    first_link_id = 0;
    last_link_id = num_of_link - 1;
    num_of_links = num_of_link;
    link_length = 1.6f;
}
