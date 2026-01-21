LIBRARY ieee;
    USE ieee.std_logic_1164.ALL;
    USE ieee.numeric_std.ALL;

ENTITY learningTop IS
    GENERIC (
        d           : integer := 1000;
        num_classes : integer := 10
    );
    PORT (
        clk                  : IN  std_logic;
        rst                  : IN  std_logic;
        correct_label        : IN  integer;                          --ground truth label
        predicted_label      : IN  integer;                          --predicted label
        similarity_correct   : IN  integer;                          --hamming distance to correct class vector
        similarity_incorrect : IN  integer;                          --hamming distance to predicted class vector
        qhv                  : IN  std_logic_vector(d - 1 DOWNTO 0); --wrongly predicted query vector
        binary_correct       : OUT std_logic_vector(d - 1 DOWNTO 0); --binarized updated correct class weights
        binary_predicted     : OUT std_logic_vector(d - 1 DOWNTO 0); --binarized updated predicted class weights
        done                 : OUT std_logic
    );
END ENTITY;

ARCHITECTURE behavioral OF learningTop IS

    FUNCTION clog2(n : integer) RETURN integer IS
        VARIABLE i : integer := 0;
        VARIABLE v : integer := n - 1;
    BEGIN
        WHILE v > 0 LOOP
            v := v / 2;
            i := i + 1;
        END LOOP;
        RETURN i;
    END FUNCTION;

    CONSTANT addr_w : integer := clog2(d * num_classes);

    COMPONENT binarizer
        GENERIC (
            d : integer := 1000
        );
        PORT (
            start         : IN  std_logic;
            int_vector    : IN  integer_vector(d - 1 DOWNTO 0);
            result_vector : OUT std_logic_vector(d - 1 DOWNTO 0)
        );
    END COMPONENT;

    COMPONENT popCount IS
        GENERIC (lenPop : INTEGER := 8); -- bit width out popCounters
        PORT (
            clk, rst : IN  STD_LOGIC;
            en       : IN  STD_LOGIC;
            dout     : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    --This or BRAM in Block Design
    TYPE shadow_vector IS ARRAY (0 TO d - 1) OF integer;
    SIGNAL update_correct   : shadow_vector;
    SIGNAL update_predicted : shadow_vector;
    SIGNAL update_completed : std_logic := '0';
    SIGNAL current_idx      : integer   := 0;

    COMPONENT blk_mem_gen_LEARN IS
        PORT (
            clka   : IN  STD_LOGIC;
            ena    : IN  STD_LOGIC;
            wea    : IN  STD_LOGIC_VECTOR (3 DOWNTO 0);
            addra  : IN  STD_LOGIC_VECTOR (addr_w - 1 DOWNTO 0);
            dina   : IN  STD_LOGIC_VECTOR (31 DOWNTO 0);
            douta  : OUT STD_LOGIC_VECTOR (31 DOWNTO 0);
            clkb   : IN  STD_LOGIC;
            enb    : IN  STD_LOGIC;
            web    : IN  STD_LOGIC_VECTOR (3 DOWNTO 0);
            addrb  : IN  STD_LOGIC_VECTOR (addr_w - 1 DOWNTO 0);
            dinb   : IN  STD_LOGIC_VECTOR (31 DOWNTO 0);
            doutb  : OUT STD_LOGIC_VECTOR (31 DOWNTO 0)
        );
    END COMPONENT;

    SIGNAL learn_addra  : std_logic_vector(addr_w - 1 DOWNTO 0);
    SIGNAL learn_addrb  : std_logic_vector(addr_w - 1 DOWNTO 0);
    SIGNAL learn_dina   : std_logic_vector(31 DOWNTO 0) := (others => '0');
    SIGNAL learn_dinb   : std_logic_vector(31 DOWNTO 0) := (others => '0');
    SIGNAL learn_douta  : std_logic_vector(31 DOWNTO 0);
    SIGNAL learn_doutb  : std_logic_vector(31 DOWNTO 0);
    SIGNAL learn_wea    : std_logic_vector(3 DOWNTO 0) := (others => '0');
    SIGNAL learn_web    : std_logic_vector(3 DOWNTO 0) := (others => '0');
    SIGNAL learn_ena    : std_logic := '1';
    SIGNAL learn_enb    : std_logic := '1';

    FUNCTION scaling(
            value         : integer;
            similarity    : integer;
            predicted_bit : std_logic;
            punish        : std_logic
        ) RETURN integer IS
        VARIABLE result : integer;
        CONSTANT lr : integer := 64;
    BEGIN
        result := d - similarity;
        IF punish = '1' XOR predicted_bit = '0' THEN
            result := result * (- 1);
        END IF;
        result := (result * lr / d) + value;
        RETURN result;
    END FUNCTION;

BEGIN

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN

            -- Ensure we only run when not finished
            IF update_completed = '0' THEN

                -- 1. PERFORM THE UPDATE (One index per cycle)
                --    We use the signal 'current_idx' instead of loop variable 'i'
                update_correct(current_idx) <= scaling(
                    to_integer(signed(learn_douta)),
                    similarity_correct,
                    qhv(current_idx),
                    '0'
                );

                update_predicted(current_idx) <= scaling(
                    to_integer(signed(learn_doutb)),
                    similarity_incorrect,
                    qhv(current_idx),
                    '1'
                );
            ELSE
                done <= '1';
                --binarize the updated weights 
            END IF;
            IF current_idx = d - 1 THEN
                update_completed <= '1';
            END IF;
        END IF;
    END PROCESS;

    PROCESS (clk)
    BEGIN
        IF rising_edge(clk) THEN
            IF rst = '1' THEN
                update_completed <= '0';
                current_idx <= 0;
                done <= '0';
            END IF;
        END IF;
    END PROCESS;

    sel: popCount
        GENERIC MAP (10)
        PORT MAP (
            clk, update_completed, NOT update_completed, current_idx
        );

    binarizeCorrect: FOR i IN 0 TO d - 1 GENERATE
        binarizer_inst: binarizer
            PORT MAP (
                update_completed,
                update_correct(i),
                binary_correct(i)
            );
    END GENERATE;

    binarizePredicted: FOR i IN 0 TO d - 1 GENERATE
        binarizer_inst: binarizer
            PORT MAP (
                update_completed,
                update_predicted(i),
                binary_predicted(i)
            );
    END GENERATE;

    learn_addra <= std_logic_vector(to_unsigned(correct_label * d + current_idx, addr_w));
    learn_addrb <= std_logic_vector(to_unsigned(predicted_label * d + current_idx, addr_w));

    learn_mem: blk_mem_gen_LEARN
        PORT MAP (
            clka  => clk,
            ena   => learn_ena,
            wea   => learn_wea,
            addra => learn_addra,
            dina  => learn_dina,
            douta => learn_douta,
            clkb  => clk,
            enb   => learn_enb,
            web   => learn_web,
            addrb => learn_addrb,
            dinb  => learn_dinb,
            doutb => learn_doutb
        );

END ARCHITECTURE;
